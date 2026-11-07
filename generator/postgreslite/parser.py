"""The SQL parser: tokens (lexer.py) -> parse trees (ast.py), after PostgreSQL 16's gram.y.

Recursive descent with gram.y's operator precedence (low to high):
  OR < AND < NOT < IS, ISNULL, NOTNULL < comparisons (< > = <= >= <>) < BETWEEN IN LIKE ILIKE SIMILAR
  < other operators (|| -> @> ~ ...) < + - < * / % < ^ < AT TIME ZONE < COLLATE < unary minus < [] < :: < .
Keywords follow kwlist.h's categories, so aliases and column names that real PostgreSQL rejects are rejected here too
("syntax error at or near ..." with the token's position).
"""

from __future__ import annotations

from . import ast as A
from .errors import PgError, Unsupported, syntax_error
from .lexer import AS_LABEL, KEYWORDS, Token, tokenize

COMPARE = {"<", ">", "=", "<=", ">=", "<>"}
NOT_QUAL = COMPARE | {"+", "-", "*", "/", "%", "^"}

# keywords that end an expression's alias-less continuation in a target list etc.
_TYPE_KW = {"int", "integer", "smallint", "bigint", "real", "float", "double", "decimal", "dec", "numeric",
            "boolean", "char", "character", "varchar", "national", "nchar", "bit", "timestamp", "time", "interval"}


def parse(sql: str) -> list:
    """Every statement in `sql` (separated by semicolons). Each statement gets .text (its source) and .start."""
    return Parser(sql).statements()


def parse_one(sql: str):
    stmts = parse(sql)
    if len(stmts) != 1:
        raise PgError("expected exactly one statement", "42601")
    return stmts[0]


def parse_expr(sql: str):
    p = Parser(sql)
    e = p.a_expr()
    p.expect_eof()
    return e


def parse_type(sql: str) -> A.TypeName:
    p = Parser(sql)
    t = p.typename()
    p.expect_eof()
    return t


class Parser:
    def __init__(self, sql: str):
        self.sql = sql
        self.toks: list[Token] = tokenize(sql)
        self.i = 0

    # ------------------------------------------------------------ token helpers
    @property
    def tok(self) -> Token:
        return self.toks[self.i]

    def peek(self, k: int = 1) -> Token:
        j = min(self.i + k, len(self.toks) - 1)
        return self.toks[j]

    def advance(self) -> Token:
        t = self.toks[self.i]
        if t.kind != "EOF":
            self.i += 1
        return t

    def error(self, tok: Token | None = None):
        t = tok or self.tok
        if t.kind == "EOF":
            return syntax_error(None, len(self.sql) + 1 if t.pos >= len(self.sql) else t.pos + 1)
        return syntax_error(t.text, t.pos + 1)

    def is_kw(self, *words) -> bool:
        return self.tok.kind == "KW" and self.tok.value in words

    def is_self(self, ch) -> bool:
        return self.tok.kind == "SELF" and self.tok.value == ch

    def is_op(self, op) -> bool:
        return self.tok.kind in ("OP", "SELF") and self.tok.value == op

    def accept_kw(self, *words) -> Token | None:
        if self.is_kw(*words):
            return self.advance()
        return None

    def accept_self(self, ch) -> Token | None:
        if self.is_self(ch):
            return self.advance()
        return None

    def expect_kw(self, *words) -> Token:
        if not self.is_kw(*words):
            raise self.error()
        return self.advance()

    def expect_self(self, ch) -> Token:
        if not self.is_self(ch):
            raise self.error()
        return self.advance()

    def expect_eof(self):
        if self.tok.kind != "EOF":
            raise self.error()

    def accept_kws(self, *words) -> bool:
        """Accept a sequence of keywords (all or nothing)."""
        for k, w in enumerate(words):
            t = self.peek(k)
            if not (t.kind == "KW" and t.value == w):
                return False
        for _ in words:
            self.advance()
        return True

    # identifier categories (gram.y)
    @staticmethod
    def _cat(t: Token) -> str | None:
        return KEYWORDS.get(t.value) if t.kind == "KW" else None

    def is_colid(self, t: Token | None = None) -> bool:
        t = t or self.tok
        return t.kind in ("IDENT", "QIDENT") or (t.kind == "KW" and self._cat(t) in ("U", "C"))

    def is_type_func_name(self, t: Token | None = None) -> bool:
        t = t or self.tok
        return t.kind in ("IDENT", "QIDENT") or (t.kind == "KW" and self._cat(t) in ("U", "T"))

    def is_col_label(self, t: Token | None = None) -> bool:
        t = t or self.tok
        return t.kind in ("IDENT", "QIDENT", "KW")

    def is_bare_label(self, t: Token | None = None) -> bool:
        t = t or self.tok
        return t.kind in ("IDENT", "QIDENT") or (t.kind == "KW" and t.value not in AS_LABEL)

    def is_nonreserved(self, t: Token | None = None) -> bool:
        t = t or self.tok
        return t.kind in ("IDENT", "QIDENT") or (t.kind == "KW" and self._cat(t) in ("U", "C", "T"))

    def colid(self) -> str:
        if not self.is_colid():
            raise self.error()
        return self.advance().value

    def col_label(self) -> str:
        if not self.is_col_label():
            raise self.error()
        return self.advance().value

    def name_list(self) -> list[str]:
        out = [self.colid()]
        while self.accept_self(","):
            out.append(self.colid())
        return out

    def paren_name_list(self) -> list[str]:
        self.expect_self("(")
        out = self.name_list()
        self.expect_self(")")
        return out

    def any_name(self) -> list[str]:
        out = [self.colid()]
        while self.is_self("."):
            self.advance()
            out.append(self.col_label())
        return out

    def qualified_name(self) -> A.RangeVar:
        pos = self.tok.pos
        names = self.any_name()
        if len(names) == 1:
            return A.RangeVar(None, names[0], pos=pos)
        if len(names) == 2:
            return A.RangeVar(names[0], names[1], pos=pos)
        if len(names) == 3:
            rv = A.RangeVar(names[1], names[2], pos=pos)
            rv.catalog = names[0]
            return rv
        raise PgError(f"improper qualified name (too many dotted names): {'.'.join(names)}", "42601", pos=pos + 1)

    def text_of(self, start: int, end: int) -> str:
        return self.sql[start:end]

    def last_end(self) -> int:
        return self.toks[self.i - 1].end if self.i > 0 else 0

    # ------------------------------------------------------------ statements
    def statements(self) -> list:
        out = []
        while True:
            while self.accept_self(";"):
                pass
            if self.tok.kind == "EOF":
                return out
            start = self.tok.pos
            st = self.statement()
            end = self.last_end()
            st.text = self.sql[start:end]
            st.start = start
            out.append(st)
            if self.tok.kind == "EOF":
                return out
            if not self.is_self(";"):
                raise self.error()

    def statement(self):
        t = self.tok
        if t.kind == "SELF" and t.value == "(":
            return self.select_stmt()
        if t.kind != "KW" and t.kind != "IDENT":
            raise self.error()
        w = t.value
        if w in ("select", "values", "with", "table"):
            if w == "with":
                return self.with_stmt()
            return self.select_stmt()
        handler = {
            "insert": self.insert_stmt, "update": self.update_stmt, "delete": self.delete_stmt,
            "merge": self.merge_stmt, "create": self.create_stmt, "alter": self.alter_stmt, "drop": self.drop_stmt,
            "truncate": self.truncate_stmt, "begin": self.txn_stmt, "start": self.txn_stmt, "commit": self.txn_stmt,
            "end": self.txn_stmt, "rollback": self.txn_stmt, "abort": self.txn_stmt, "savepoint": self.txn_stmt,
            "release": self.txn_stmt, "explain": self.explain_stmt, "analyze": self.analyze_stmt,
            "analyse": self.analyze_stmt, "vacuum": self.analyze_stmt, "set": self.set_stmt, "reset": self.reset_stmt,
            "show": self.show_stmt, "copy": self.copy_stmt, "comment": self.comment_stmt,
            "prepare": self.prepare_stmt, "execute": self.execute_stmt, "deallocate": self.deallocate_stmt,
            "grant": self.grant_stmt, "revoke": self.grant_stmt, "refresh": self.refresh_stmt,
        }.get(w)
        if handler is not None:
            return handler()
        if w in ("listen", "notify", "unlisten", "discard", "checkpoint", "load", "lock", "cluster", "reindex"):
            pos = t.pos
            tag = {"lock": "LOCK TABLE"}.get(w, w.upper())
            if w == "discard":
                self.advance()
                tag = "DISCARD " + self.advance().text.upper()
            while self.tok.kind != "EOF" and not self.is_self(";"):
                self.advance()
            return A.Noop(tag, pos=pos)
        if w == "do":
            raise Unsupported("DO blocks (procedural code)", t.pos + 1)
        if w == "call":
            raise Unsupported("CALL", t.pos + 1)
        if w in ("declare", "fetch", "move", "close"):
            raise Unsupported("cursors", t.pos + 1)
        if w in ("security", "import", "reassign", "import"):
            raise Unsupported(w.upper(), t.pos + 1)
        raise self.error()

    # ------------------------------------------------------------ SELECT
    def with_stmt(self):
        """WITH ... followed by SELECT / INSERT / UPDATE / DELETE / MERGE."""
        w = self.with_clause()
        if self.is_kw("insert"):
            st = self.insert_stmt()
        elif self.is_kw("update"):
            st = self.update_stmt()
        elif self.is_kw("delete"):
            st = self.delete_stmt()
        elif self.is_kw("merge"):
            st = self.merge_stmt()
        else:
            st = self.select_no_with()
            if st.with_ is not None:
                raise PgError("multiple WITH clauses not allowed", "42601", pos=st.with_.pos + 1)
        st.with_ = w
        return st

    def with_clause(self) -> A.With:
        pos = self.expect_kw("with").pos
        rec = bool(self.accept_kw("recursive"))
        ctes = [self.cte()]
        while self.accept_self(","):
            ctes.append(self.cte())
        return A.With(ctes, rec, pos=pos)

    def cte(self) -> A.CTE:
        pos = self.tok.pos
        name = self.colid()
        cols = self.paren_name_list() if self.is_self("(") else []
        self.expect_kw("as")
        mat = None
        if self.accept_kw("materialized"):
            mat = True
        elif self.is_kw("not") and self.peek().is_kw("materialized"):
            self.advance()
            self.advance()
            mat = False
        self.expect_self("(")
        t = self.tok
        if t.is_kw("insert"):
            q = self.insert_stmt()
        elif t.is_kw("update"):
            q = self.update_stmt()
        elif t.is_kw("delete"):
            q = self.delete_stmt()
        else:
            q = self.select_stmt()
        self.expect_self(")")
        c = A.CTE(name, cols, q, mat, pos=pos)
        if self.is_kw("search"):
            raise Unsupported("SEARCH clauses of recursive queries", self.tok.pos + 1)
        if self.is_kw("cycle"):
            raise Unsupported("CYCLE clauses of recursive queries", self.tok.pos + 1)
        return c

    def select_stmt(self):
        """A full SELECT: [WITH] select_clause [ORDER BY] [LIMIT/OFFSET/FETCH] [FOR UPDATE]."""
        if self.is_kw("with"):
            w = self.with_clause()
            st = self.select_no_with()
            if st.with_ is not None:
                raise PgError("multiple WITH clauses not allowed", "42601", pos=st.with_.pos + 1)
            st.with_ = w
            return st
        return self.select_no_with()

    def select_no_with(self):
        node = self.select_clause()
        self.select_tail(node)
        return node

    def select_tail(self, node):
        if self.is_kw("order"):
            pos = self.tok.pos
            order = self.sort_clause()
            if node.order_by:
                raise PgError("multiple ORDER BY clauses not allowed", "42601", pos=pos + 1)
            node.order_by = order
        self.limit_clauses(node)
        if self.is_kw("for"):
            node.locking = self.locking_clause()
            self.limit_clauses(node)

    def limit_clauses(self, node):
        seen_limit = node.limit is not None
        seen_offset = node.offset is not None
        while True:
            t = self.tok
            if t.is_kw("limit"):
                if seen_limit:
                    raise PgError("multiple LIMIT clauses not allowed", "42601", pos=t.pos + 1)
                self.advance()
                if self.accept_kw("all"):
                    node.limit = A.Const(None, "null", pos=t.pos)
                else:
                    node.limit = self.a_expr()
                if self.is_self(","):
                    raise PgError("LIMIT #,# syntax is not supported", "42601", pos=self.tok.pos + 1,
                                  hint="Use separate LIMIT and OFFSET clauses.")
                seen_limit = True
            elif t.is_kw("offset"):
                if seen_offset:
                    raise PgError("multiple OFFSET clauses not allowed", "42601", pos=t.pos + 1)
                self.advance()
                node.offset = self.a_expr()
                self.accept_kw("row", "rows")
                seen_offset = True
            elif t.is_kw("fetch"):
                if seen_limit:
                    raise PgError("multiple LIMIT clauses not allowed", "42601", pos=t.pos + 1)
                self.advance()
                self.expect_kw("first", "next")
                if self.is_kw("row", "rows"):
                    cnt = A.Const(1, "int", pos=self.tok.pos)
                else:
                    cnt = self.c_or_signed()
                self.expect_kw("row", "rows")
                if self.accept_kw("only"):
                    pass
                elif self.accept_kws("with", "ties"):
                    node.with_ties = True
                else:
                    raise self.error()
                node.limit = cnt
                node.fetch_first = True
                seen_limit = True
            else:
                return

    def c_or_signed(self):
        """FETCH FIRST's count: a constant, a signed constant, or a parenthesized expression."""
        if self.is_self("("):
            return self.primary()
        if self.is_op("-") or self.is_op("+"):
            neg = self.advance().value == "-"
            e = self.primary()
            return self.negate(e, self.toks[self.i - 1].pos) if neg else e
        return self.primary()

    def locking_clause(self):
        out = []
        while self.accept_kw("for"):
            pos = self.tok.pos
            if self.accept_kw("update"):
                strength = "update"
            elif self.accept_kw("share"):
                strength = "share"
            elif self.accept_kws("no", "key", "update"):
                strength = "no key update"
            elif self.accept_kws("key", "share"):
                strength = "key share"
            else:
                raise self.error()
            rels = []
            if self.accept_kw("of"):
                rels.append(self.qualified_name())
                while self.accept_self(","):
                    rels.append(self.qualified_name())
            if self.accept_kw("nowait"):
                pass
            elif self.accept_kws("skip", "locked"):
                pass
            out.append((strength, rels, pos))
        return out

    def select_clause(self):
        """Set operations: INTERSECT binds tighter than UNION / EXCEPT (both left-associative)."""
        left = self.select_intersect()
        while self.is_kw("union", "except"):
            t = self.advance()
            all_ = self.set_quantifier()
            right = self.select_intersect()
            left = A.SetOp(t.value, all_, left, right, pos=t.pos)
        return left

    def select_intersect(self):
        left = self.select_primary()
        while self.is_kw("intersect"):
            t = self.advance()
            all_ = self.set_quantifier()
            right = self.select_primary()
            left = A.SetOp("intersect", all_, left, right, pos=t.pos)
        return left

    def set_quantifier(self) -> bool:
        if self.accept_kw("all"):
            return True
        self.accept_kw("distinct")
        return False

    def select_primary(self):
        t = self.tok
        if t.is_self("("):
            self.advance()
            inner = self.select_stmt()
            self.expect_self(")")
            inner.parens = True
            return inner
        if t.is_kw("select"):
            return self.simple_select()
        if t.is_kw("values"):
            return self.values_clause()
        if t.is_kw("table"):
            self.advance()
            rv = self.relation_expr()
            return A.Select(targets=[A.ResTarget(A.Star(pos=t.pos), None, pos=t.pos)], from_=[rv], pos=t.pos)
        raise self.error()

    def values_clause(self, allow_default: bool = False) -> A.Values:
        pos = self.expect_kw("values").pos
        rows = [self.values_row()]
        while self.accept_self(","):
            rows.append(self.values_row())
        return A.Values(rows, pos=pos)

    def values_row(self) -> list:
        self.expect_self("(")
        row = [self.expr_or_default()]
        while self.accept_self(","):
            row.append(self.expr_or_default())
        self.expect_self(")")
        return row

    def expr_or_default(self):
        if self.is_kw("default"):
            return A.SetToDefault(pos=self.advance().pos)
        return self.a_expr()

    def simple_select(self) -> A.Select:
        pos = self.expect_kw("select").pos
        s = A.Select(pos=pos)
        if self.accept_kw("all"):
            pass
        elif self.is_kw("distinct"):
            self.advance()
            if self.accept_kw("on"):
                self.expect_self("(")
                s.distinct = [self.a_expr()]
                while self.accept_self(","):
                    s.distinct.append(self.a_expr())
                self.expect_self(")")
            else:
                s.distinct = True
        s.targets = self.target_list(optional=True)
        if self.is_kw("into"):
            self.advance()
            temp = bool(self.accept_kw("temp", "temporary"))
            if not temp and self.accept_kw("unlogged"):
                pass
            self.accept_kw("table")
            s.into = (self.qualified_name(), temp)
        if self.accept_kw("from"):
            s.from_ = self.from_list()
        if self.accept_kw("where"):
            s.where = self.a_expr()
        if self.is_kw("group"):
            self.advance()
            self.expect_kw("by")
            if self.accept_kw("distinct"):
                s.group_distinct = True
            else:
                self.accept_kw("all")
            s.group_by = [self.group_item()]
            while self.accept_self(","):
                s.group_by.append(self.group_item())
        if self.accept_kw("having"):
            s.having = self.a_expr()
        if self.accept_kw("window"):
            s.windows = [self.window_definition()]
            while self.accept_self(","):
                s.windows.append(self.window_definition())
        if self.is_kw("qualify") or (self.tok.kind == "IDENT" and self.tok.value == "qualify"):
            raise self.error()
        return s

    def target_list(self, optional: bool = False) -> list:
        enders = ("from", "where", "group", "having", "window", "order", "limit", "offset", "fetch", "for", "into",
                  "union", "intersect", "except", "returning", "on")
        if optional and (self.tok.kind == "EOF" or self.is_self(";") or self.is_self(")") or self.is_kw(*enders)):
            return []
        out = [self.target_el()]
        while self.accept_self(","):
            out.append(self.target_el())
        return out

    def target_el(self) -> A.ResTarget:
        t = self.tok
        if t.is_self("*"):
            self.advance()
            return A.ResTarget(A.Star(pos=t.pos), None, pos=t.pos)
        e = self.a_expr()
        name = None
        if self.accept_kw("as"):
            name = self.col_label()
        elif self.is_bare_label() and not self.is_kw(*_CLAUSE_STARTERS):
            name = self.advance().value
        return A.ResTarget(e, name, pos=t.pos)

    def group_item(self):
        t = self.tok
        if t.is_self("(") and self.peek().is_self(")"):
            self.advance()
            self.advance()
            return A.GroupingSet("empty", [], pos=t.pos)
        if t.is_kw("rollup", "cube") and self.peek().is_self("("):
            self.advance()
            self.expect_self("(")
            items = [self.a_expr()]
            while self.accept_self(","):
                items.append(self.a_expr())
            self.expect_self(")")
            return A.GroupingSet(t.value, items, pos=t.pos)
        if t.is_kw("grouping") and self.peek().is_kw("sets"):
            self.advance()
            self.advance()
            self.expect_self("(")
            items = [self.group_item()]
            while self.accept_self(","):
                items.append(self.group_item())
            self.expect_self(")")
            return A.GroupingSet("sets", items, pos=t.pos)
        return self.a_expr()

    def sort_clause(self) -> list:
        self.expect_kw("order")
        self.expect_kw("by")
        out = [self.sortby()]
        while self.accept_self(","):
            out.append(self.sortby())
        return out

    def sortby(self) -> A.SortBy:
        pos = self.tok.pos
        e = self.a_expr()
        sb = A.SortBy(e, pos=pos)
        if self.accept_kw("asc"):
            pass
        elif self.accept_kw("desc"):
            sb.desc = True
        elif self.accept_kw("using"):
            op = self.advance()
            if op.kind not in ("OP", "SELF"):
                raise self.error(op)
            sb.using = op.value
            sb.desc = op.value in (">", ">=")
        if self.accept_kw("nulls"):
            w = self.expect_kw("first", "last")
            sb.nulls = w.value
        return sb

    # ------------------------------------------------------------ windows
    def window_definition(self) -> A.WindowDef:
        pos = self.tok.pos
        name = self.colid()
        self.expect_kw("as")
        w = self.window_spec()
        w.name = name
        w.pos = pos
        return w

    def window_spec(self) -> A.WindowDef:
        pos = self.expect_self("(").pos
        w = A.WindowDef(pos=pos)
        if self.is_colid() and not self.is_kw("partition", "range", "rows", "groups", "order"):
            w.ref = self.colid()
        if self.accept_kw("partition"):
            self.expect_kw("by")
            w.partition = [self.a_expr()]
            while self.accept_self(","):
                w.partition.append(self.a_expr())
        if self.is_kw("order"):
            w.order = self.sort_clause()
        if self.is_kw("range", "rows", "groups"):
            w.frame_mode = self.advance().value
            if self.accept_kw("between"):
                w.start = self.frame_bound()
                self.expect_kw("and")
                w.end = self.frame_bound()
            else:
                w.start = self.frame_bound()
                w.end = None
            if self.accept_kw("exclude"):
                if self.accept_kws("current", "row"):
                    w.exclude = "current_row"
                elif self.accept_kw("group"):
                    w.exclude = "group"
                elif self.accept_kw("ties"):
                    w.exclude = "ties"
                elif self.accept_kws("no", "others"):
                    w.exclude = None
                else:
                    raise self.error()
        self.expect_self(")")
        return w

    def frame_bound(self) -> A.FrameBound:
        t = self.tok
        if t.is_kw("unbounded"):
            self.advance()
            k = self.expect_kw("preceding", "following")
            return A.FrameBound("unbounded_" + k.value, pos=t.pos)
        if t.is_kw("current"):
            self.advance()
            self.expect_kw("row")
            return A.FrameBound("current_row", pos=t.pos)
        off = self.a_expr()
        k = self.expect_kw("preceding", "following")
        return A.FrameBound(k.value, off, pos=t.pos)

    # ------------------------------------------------------------ FROM
    def from_list(self) -> list:
        out = [self.table_ref()]
        while self.accept_self(","):
            out.append(self.table_ref())
        return out

    def table_ref(self):
        left = self.table_primary()
        while True:
            t = self.tok
            if t.is_kw("cross") and self.peek().is_kw("join"):
                self.advance()
                self.advance()
                right = self.table_primary()
                left = A.JoinExpr("cross", left, right, pos=t.pos)
                continue
            natural = False
            if t.is_kw("natural"):
                natural = True
                self.advance()
            kind = None
            if self.is_kw("join"):
                kind = "inner"
            elif self.is_kw("inner"):
                self.advance()
                kind = "inner"
            elif self.is_kw("left", "right", "full"):
                kind = self.advance().value
                self.accept_kw("outer")
            if kind is None:
                if natural:
                    raise self.error()
                return left
            self.expect_kw("join")
            right = self.table_ref_for_join()
            j = A.JoinExpr(kind, left, right, natural=natural, pos=t.pos)
            if not natural:
                if self.accept_kw("on"):
                    j.quals = self.a_expr()
                elif self.accept_kw("using"):
                    j.using = self.paren_name_list()
                    if self.accept_kw("as"):
                        j.using_alias = self.colid()
                else:
                    raise self.error()
            left = j

    def table_ref_for_join(self):
        """The right side of JOIN: a primary, or a nested join that has its own ON (a JOIN b JOIN c ON .. ON ..)."""
        right = self.table_primary()
        t = self.tok
        if t.is_kw("join", "inner", "left", "right", "full", "natural", "cross"):
            # only nest when the right side's own join is followed by its qualification before ours
            save = self.i
            try:
                inner = self._join_onto(right)
                if self.is_kw("on", "using"):
                    return inner
            except PgError:
                pass
            self.i = save
        return right

    def _join_onto(self, left):
        t = self.tok
        if t.is_kw("cross"):
            self.advance()
            self.expect_kw("join")
            return A.JoinExpr("cross", left, self.table_primary(), pos=t.pos)
        natural = bool(self.accept_kw("natural"))
        kind = "inner"
        if self.accept_kw("inner"):
            pass
        elif self.is_kw("left", "right", "full"):
            kind = self.advance().value
            self.accept_kw("outer")
        self.expect_kw("join")
        right = self.table_ref_for_join()
        j = A.JoinExpr(kind, left, right, natural=natural, pos=t.pos)
        if not natural:
            if self.accept_kw("on"):
                j.quals = self.a_expr()
            elif self.accept_kw("using"):
                j.using = self.paren_name_list()
            else:
                raise self.error()
        return j

    def table_primary(self):
        t = self.tok
        lateral = False
        if t.is_kw("lateral"):
            lateral = True
            self.advance()
            t = self.tok
        if t.is_self("("):
            if self._starts_select(self.i + 1):
                save = self.i
                try:
                    self.advance()
                    q = self.select_stmt()
                    self.expect_self(")")
                except PgError:
                    self.i = save
                    q = None
                if q is not None:
                    alias = self.opt_alias()
                    if alias is None:
                        pass   # PG 16 allows a subquery in FROM without an alias
                    return A.RangeSubselect(q, alias, lateral, pos=t.pos)
            if lateral:
                raise self.error()
            self.advance()
            inner = self.table_ref()
            self.expect_self(")")
            alias = self.opt_alias()
            if alias is not None:
                if isinstance(inner, A.JoinExpr):
                    inner.alias = alias
                elif isinstance(inner, (A.RangeVar, A.RangeSubselect, A.RangeFunction)):
                    inner.alias = alias
            return inner
        if t.is_kw("rows") and self.peek().is_kw("from"):
            self.advance()
            self.advance()
            self.expect_self("(")
            funcs = [self.func_in_from()]
            while self.accept_self(","):
                funcs.append(self.func_in_from())
            self.expect_self(")")
            return self.func_table_tail(funcs, lateral, t.pos)
        if t.is_kw("only"):
            rv = self.relation_expr()
            rv.alias = self.opt_alias()
            return rv
        # a function call in FROM, or a table
        if self._is_func_start():
            f = self.func_in_from()
            return self.func_table_tail([f], lateral, t.pos)
        if lateral:
            raise self.error()
        rv = self.relation_expr()
        rv.alias = self.opt_alias()
        if self.is_kw("tablesample"):
            self.advance()
            method = self.colid()
            self.expect_self("(")
            args = [self.a_expr()]
            while self.accept_self(","):
                args.append(self.a_expr())
            self.expect_self(")")
            seed = None
            if self.accept_kw("repeatable"):
                self.expect_self("(")
                seed = self.a_expr()
                self.expect_self(")")
            rv.sample = (method, args, seed)
        return rv

    def func_in_from(self):
        e = self.primary()
        if not isinstance(e, (A.FuncCall, A.SQLValue)):
            raise self.error()
        return e

    def func_table_tail(self, funcs, lateral, pos):
        ordinality = False
        if self.is_kw("with") and self.peek().is_kw("ordinality"):
            self.advance()
            self.advance()
            ordinality = True
        rf = A.RangeFunction(funcs, None, lateral, ordinality, pos=pos)
        # alias with column names, or a column definition list (jsonb_to_record ... AS x(a int, b text))
        has_as = bool(self.accept_kw("as"))
        if has_as or self.is_colid():
            apos = self.tok.pos
            if has_as and self.is_self("("):
                rf.coldefs = self.coldef_list()
                return rf
            name = self.colid()
            alias = A.Alias(name, pos=apos)
            if self.is_self("("):
                # names, or name type pairs
                save = self.i
                self.advance()
                if self.is_colid() and (self.peek().is_self(",") or self.peek().is_self(")")):
                    self.i = save
                    alias.cols = self.paren_name_list()
                else:
                    self.i = save
                    rf.coldefs = self.coldef_list()
            rf.alias = alias
        elif self.is_self("("):
            rf.coldefs = self.coldef_list()
        return rf

    def coldef_list(self):
        self.expect_self("(")
        out = []
        while True:
            pos = self.tok.pos
            name = self.colid()
            typ = self.typename()
            out.append(A.ColumnDef(name, typ, [], pos=pos))
            if not self.accept_self(","):
                break
        self.expect_self(")")
        return out

    def _is_func_start(self) -> bool:
        """name ( or schema.name ( at the current token."""
        t = self.tok
        if not (self.is_type_func_name(t) or self.is_colid(t)):
            if t.kind == "KW" and t.value in ("coalesce", "greatest", "least", "nullif", "extract", "substring",
                                               "trim", "position", "overlay", "current_date", "current_timestamp",
                                               "localtimestamp", "current_time", "localtime", "current_user",
                                               "session_user", "user", "current_role", "current_schema",
                                               "current_catalog", "system_user", "normalize", "xmltable"):
                return True
            return False
        j = self.i + 1
        while self.toks[j].is_self(".") and self.toks[j + 1].kind in ("IDENT", "QIDENT", "KW"):
            j += 2
        return self.toks[j].is_self("(")

    def _starts_select(self, j: int) -> bool:
        while self.toks[j].is_self("("):
            j += 1
        t = self.toks[j]
        return t.kind == "KW" and t.value in ("select", "values", "with", "table")

    def relation_expr(self) -> A.RangeVar:
        only = False
        if self.accept_kw("only"):
            only = True
            if self.accept_self("("):
                rv = self.qualified_name()
                self.expect_self(")")
                rv.only = True
                return rv
        rv = self.qualified_name()
        rv.only = only
        self.accept_self("*")
        return rv

    def opt_alias(self, allow_set: bool = True) -> A.Alias | None:
        pos = self.tok.pos
        if self.accept_kw("as"):
            name = self.colid()
        elif self.is_colid() and not (not allow_set and self.is_kw("set")):
            name = self.advance().value
        else:
            return None
        a = A.Alias(name, pos=pos)
        if self.is_self("("):
            a.cols = self.paren_name_list()
        return a

    # ------------------------------------------------------------ expressions
    def a_expr(self):
        return self.or_expr()

    def or_expr(self):
        left = self.and_expr()
        if not self.is_kw("or"):
            return left
        args = [left]
        pos = self.tok.pos
        while self.accept_kw("or"):
            args.append(self.and_expr())
        return A.BoolOp("or", args, pos=pos)

    def and_expr(self):
        left = self.not_expr()
        if not self.is_kw("and"):
            return left
        args = [left]
        pos = self.tok.pos
        while self.accept_kw("and"):
            args.append(self.not_expr())
        return A.BoolOp("and", args, pos=pos)

    def not_expr(self):
        t = self.tok
        if t.is_kw("not"):
            self.advance()
            return A.BoolOp("not", [self.not_expr()], pos=t.pos)
        return self.is_expr()

    def is_expr(self, bexpr: bool = False):
        e = self.cmp_expr(bexpr)
        while True:
            t = self.tok
            if t.is_kw("is"):
                self.advance()
                neg = bool(self.accept_kw("not"))
                if self.accept_kw("null"):
                    e = A.NullTest(e, neg, pos=t.pos)
                elif self.is_kw("true", "false", "unknown"):
                    e = A.BoolTest(e, self.advance().value, neg, pos=t.pos)
                elif self.accept_kw("distinct"):
                    self.expect_kw("from")
                    r = self.cmp_expr(bexpr)
                    e = A.Distinct(e, r, neg, pos=t.pos)
                elif self.is_kw("json"):
                    raise Unsupported("IS JSON", t.pos + 1)
                elif self.is_kw("normalized", "nfc", "nfd", "nfkc", "nfkd", "document", "of"):
                    raise Unsupported("IS " + self.tok.text.upper(), t.pos + 1)
                else:
                    raise self.error()
            elif t.is_kw("isnull"):
                self.advance()
                e = A.NullTest(e, False, pos=t.pos)
            elif t.is_kw("notnull"):
                self.advance()
                e = A.NullTest(e, True, pos=t.pos)
            else:
                return e

    def cmp_expr(self, bexpr: bool = False):
        left = self.like_expr(bexpr)
        t = self.tok
        if t.kind in ("OP", "SELF") and t.value in COMPARE:
            self.advance()
            right = self.sub_or(lambda: self.like_expr(bexpr), t.value, left, t.pos)
            if isinstance(right, A.Node) and getattr(right, "_done", False):
                return right
            e = A.Op(t.value, left, right, pos=t.pos)
            t2 = self.tok
            if t2.kind in ("OP", "SELF") and t2.value in COMPARE:
                raise self.error(t2)       # comparisons don't chain (nonassoc)
            return e
        return left

    def sub_or(self, parse_right, op, left, pos):
        """After an operator: ANY / SOME / ALL (subquery or array), else the right operand."""
        if self.is_kw("any", "some", "all") and self.peek().is_self("("):
            kind = "all" if self.advance().value == "all" else "any"
            lp = self.tok.pos
            self.expect_self("(")
            if self._starts_select(self.i):
                q = self.select_stmt()
                self.expect_self(")")
                node = A.SubLink(kind, q, left, op, pos=pos)
            else:
                arr = self.a_expr()
                self.expect_self(")")
                node = A.AnyAll(kind, op, left, arr, pos=pos)
            node._done = True
            return node
        return parse_right()

    def like_expr(self, bexpr: bool = False):
        e = self.op_expr()
        if bexpr:
            return e
        while True:
            t = self.tok
            neg = False
            if t.is_kw("not") and self.peek().is_kw("between", "in", "like", "ilike", "similar"):
                self.advance()
                neg = True
            t2 = self.tok
            if t2.is_kw("between"):
                self.advance()
                sym = False
                if self.accept_kw("symmetric"):
                    sym = True
                else:
                    self.accept_kw("asymmetric")
                lo = self.op_expr()
                self.expect_kw("and")
                hi = self.op_expr()
                e = A.Between(e, lo, hi, neg, sym, pos=t.pos)
            elif t2.is_kw("in"):
                self.advance()
                lp = self.tok
                self.expect_self("(")
                if self._starts_select(self.i):
                    save = self.i
                    try:
                        q = self.select_stmt()
                        self.expect_self(")")
                        sl = A.SubLink("any", q, e, "=", pos=t2.pos)
                        e = A.BoolOp("not", [sl], pos=t.pos) if neg else sl
                        if neg:
                            sl.negated_in = True
                        continue
                    except PgError:
                        self.i = save
                items = [self.a_expr()]
                while self.accept_self(","):
                    items.append(self.a_expr())
                self.expect_self(")")
                e = A.InList(e, items, neg, pos=t2.pos)
            elif t2.is_kw("like", "ilike"):
                self.advance()
                kind = t2.value
                if self.is_kw("any", "some", "all") and self.peek().is_self("("):
                    op = {("like", False): "~~", ("like", True): "!~~", ("ilike", False): "~~*",
                          ("ilike", True): "!~~*"}[(kind, neg)]
                    e = self.sub_or(None, op, e, t2.pos)
                    continue
                pat = self.op_expr()
                esc = None
                if self.accept_kw("escape"):
                    esc = self.op_expr()
                e = A.Like(e, pat, esc, neg, kind, pos=t2.pos if not neg else t.pos)
            elif t2.is_kw("similar"):
                self.advance()
                self.expect_kw("to")
                pat = self.op_expr()
                esc = None
                if self.accept_kw("escape"):
                    esc = self.op_expr()
                e = A.Like(e, pat, esc, neg, "similar", pos=t2.pos)
            else:
                if neg:
                    raise self.error()
                return e

    def op_expr(self):
        """Other operators (|| -> ->> #> @> <@ ? ~ ~* !~ && ...): left-associative, prefix forms too."""
        t = self.tok
        if t.kind == "OP" and t.value not in NOT_QUAL:
            self.advance()
            operand = self.op_expr_operand()
            left = A.Op(t.value, None, operand, pos=t.pos)
        else:
            left = self.add_expr()
        while True:
            t = self.tok
            if t.kind == "OP" and t.value not in NOT_QUAL:
                self.advance()
                right = self.sub_or(self.add_expr, t.value, left, t.pos)
                if getattr(right, "_done", False):
                    left = right
                    continue
                left = A.Op(t.value, left, right, pos=t.pos)
            elif t.is_kw("operator") and self.peek().is_self("("):
                self.advance()
                self.expect_self("(")
                names = []
                while not self.is_self(")"):
                    names.append(self.advance().value)
                self.expect_self(")")
                op = names[-1]
                right = self.add_expr()
                left = A.Op(op, left, right, pos=t.pos)
            else:
                return left

    def op_expr_operand(self):
        t = self.tok
        if t.kind == "OP" and t.value not in NOT_QUAL:
            self.advance()
            return A.Op(t.value, None, self.op_expr_operand(), pos=t.pos)
        return self.add_expr()

    def add_expr(self):
        left = self.mul_expr()
        while self.tok.kind == "SELF" and self.tok.value in ("+", "-"):
            t = self.advance()
            right = self.sub_or(self.mul_expr, t.value, left, t.pos)
            left = right if getattr(right, "_done", False) else A.Op(t.value, left, right, pos=t.pos)
        return left

    def mul_expr(self):
        left = self.pow_expr()
        while self.tok.kind == "SELF" and self.tok.value in ("*", "/", "%"):
            t = self.advance()
            right = self.sub_or(self.pow_expr, t.value, left, t.pos)
            left = right if getattr(right, "_done", False) else A.Op(t.value, left, right, pos=t.pos)
        return left

    def pow_expr(self):
        left = self.at_expr()
        while self.is_self("^"):
            t = self.advance()
            right = self.sub_or(self.at_expr, t.value, left, t.pos)
            left = right if getattr(right, "_done", False) else A.Op("^", left, right, pos=t.pos)
        return left

    def at_expr(self):
        e = self.collate_expr()
        while self.is_kw("at"):
            t = self.tok
            if self.peek().is_kw("time") and self.peek(2).is_kw("zone"):
                self.advance()
                self.advance()
                self.advance()
                zone = self.collate_expr()
                e = A.AtTimeZone(e, zone, pos=t.pos)
            elif self.peek().is_kw("local"):
                self.advance()
                self.advance()
                e = A.AtTimeZone(e, None, pos=t.pos)
            else:
                return e
        return e

    def collate_expr(self):
        e = self.unary_expr()
        while self.is_kw("collate"):
            t = self.advance()
            e = A.Collate(e, self.any_name(), pos=t.pos)
        return e

    def unary_expr(self):
        t = self.tok
        if t.kind == "SELF" and t.value in ("-", "+"):
            self.advance()
            operand = self.unary_expr()
            if t.value == "-":
                return self.negate(operand, t.pos)
            return A.Op("+", None, operand, pos=t.pos)
        return self.postfix_expr()

    @staticmethod
    def negate(e, pos):
        """doNegate: fold a minus into a numeric constant."""
        if isinstance(e, A.Const) and e.kind in ("int", "numeric") and not getattr(e, "_negated", False):
            if e.kind == "int":
                c = A.Const(-int(e.value), "int", pos=pos)
            else:
                v = e.value
                c = A.Const(v[1:] if v.startswith("-") else "-" + v, "numeric", pos=pos)
            c._negated = True
            return c
        return A.Op("-", None, e, pos=pos)

    def postfix_expr(self):
        e = self.primary()
        while True:
            t = self.tok
            if t.kind == "OP" and t.value == "::":
                self.advance()
                typ = self.typename()
                e = A.Cast(e, typ, pos=t.pos)
            elif t.is_self("["):
                e = self.subscript(e)
            else:
                return e

    def subscript(self, e):
        t = self.expect_self("[")
        lo = hi = None
        is_slice = False
        if not self.is_self(":"):
            lo = self.a_expr()
        if self.accept_self(":"):
            is_slice = True
            if not self.is_self("]"):
                hi = self.a_expr()
        self.expect_self("]")
        return A.Subscript(e, lo, hi, is_slice, pos=t.pos)

    def indirection(self, e):
        """After a parenthesized expression: .field, .*, [subscripts]."""
        while True:
            t = self.tok
            if t.is_self("."):
                self.advance()
                if self.accept_self("*"):
                    e = A.FieldSel(e, "*", pos=t.pos)
                else:
                    e = A.FieldSel(e, self.col_label(), pos=t.pos)
            elif t.is_self("["):
                e = self.subscript(e)
            else:
                return e

    def primary(self):
        t = self.tok
        k = t.kind
        if k == "INT":
            self.advance()
            return A.Const(int(t.value), "int", pos=t.pos)
        if k == "NUM":
            self.advance()
            return A.Const(t.value, "numeric", pos=t.pos)
        if k == "STR":
            self.advance()
            return A.Const(t.value, "str", pos=t.pos)
        if k == "BITSTR":
            self.advance()
            return A.Const(t.value, "bitstr", pos=t.pos)
        if k == "PARAM":
            self.advance()
            e = A.Param(t.value, pos=t.pos)
            return self.indirection(e)
        if t.is_self("("):
            return self.paren_expr()
        if k == "KW":
            r = self.keyword_primary(t)
            if r is not None:
                return r
        if k in ("IDENT", "QIDENT") or (k == "KW" and self._cat(t) in ("U", "C", "T")):
            return self.name_primary()
        raise self.error()

    def paren_expr(self):
        t = self.expect_self("(")
        if self._starts_select(self.i):
            save = self.i
            try:
                q = self.select_stmt()
                self.expect_self(")")
                return self.indirection(A.SubLink("expr", q, pos=t.pos))
            except PgError:
                self.i = save
        e = self.a_expr()
        if self.is_self(","):
            items = [e]
            while self.accept_self(","):
                items.append(self.a_expr())
            self.expect_self(")")
            return A.RowExpr(items, pos=t.pos)
        self.expect_self(")")
        if self.is_self(".") or self.is_self("["):
            return self.indirection(e)
        e = _Paren.wrap(e)
        return e

    def keyword_primary(self, t):
        w = t.value
        if w == "true" or w == "false":
            self.advance()
            return A.Const(w == "true", "bool", pos=t.pos)
        if w == "null":
            self.advance()
            return A.Const(None, "null", pos=t.pos)
        if w == "case":
            return self.case_expr()
        if w == "cast":
            self.advance()
            self.expect_self("(")
            e = self.a_expr()
            self.expect_kw("as")
            typ = self.typename()
            self.expect_self(")")
            return A.Cast(e, typ, pos=t.pos, style="CAST")
        if w == "exists" and self.peek().is_self("("):
            self.advance()
            self.expect_self("(")
            q = self.select_stmt()
            self.expect_self(")")
            return A.SubLink("exists", q, pos=t.pos)
        if w == "array":
            self.advance()
            if self.is_self("("):
                self.advance()
                q = self.select_stmt()
                self.expect_self(")")
                return A.SubLink("array", q, pos=t.pos)
            return self.indirection(self.array_literal(t.pos))
        if w == "row" and self.peek().is_self("("):
            self.advance()
            self.expect_self("(")
            items = []
            if not self.is_self(")"):
                items.append(self.a_expr())
                while self.accept_self(","):
                    items.append(self.a_expr())
            self.expect_self(")")
            return A.RowExpr(items, pos=t.pos, explicit=True)
        if w in ("current_date", "current_role", "current_user", "session_user", "user", "current_catalog",
                 "system_user"):
            self.advance()
            return A.SQLValue(w, pos=t.pos)
        if w == "current_schema":
            self.advance()
            if self.accept_self("("):
                self.expect_self(")")
            return A.SQLValue(w, pos=t.pos)
        if w in ("current_time", "current_timestamp", "localtime", "localtimestamp"):
            self.advance()
            prec = None
            if self.accept_self("("):
                prec = int(self.advance().value)
                self.expect_self(")")
            return A.SQLValue(w, prec, pos=t.pos)
        if w in ("coalesce", "greatest", "least", "nullif") and self.peek().is_self("("):
            self.advance()
            self.expect_self("(")
            args = [self.a_expr()]
            while self.accept_self(","):
                args.append(self.a_expr())
            self.expect_self(")")
            return A.FuncCall([w], args, pos=t.pos, special=w)
        if w == "extract" and self.peek().is_self("("):
            self.advance()
            self.expect_self("(")
            ft = self.tok
            if ft.kind == "STR":
                field = ft.value
                self.advance()
            elif self.is_col_label():
                field = self.advance().value
            else:
                raise self.error()
            self.expect_kw("from")
            e = self.a_expr()
            self.expect_self(")")
            return A.FuncCall(["extract"], [A.Const(field, "str", pos=ft.pos), e], pos=t.pos, special="extract")
        if w == "position" and self.peek().is_self("("):
            self.advance()
            self.expect_self("(")
            a = self.op_expr()
            self.expect_kw("in")
            b = self.op_expr()
            self.expect_self(")")
            return A.FuncCall(["position"], [b, a], pos=t.pos, special="position")
        if w == "substring" and self.peek().is_self("("):
            return self.substring_expr()
        if w == "overlay" and self.peek().is_self("("):
            self.advance()
            self.expect_self("(")
            a = self.a_expr()
            self.expect_kw("placing")
            b = self.a_expr()
            self.expect_kw("from")
            c = self.a_expr()
            args = [a, b, c]
            if self.accept_kw("for"):
                args.append(self.a_expr())
            self.expect_self(")")
            return A.FuncCall(["overlay"], args, pos=t.pos, special="overlay")
        if w == "trim" and self.peek().is_self("("):
            return self.trim_expr()
        if w == "normalize" and self.peek().is_self("("):
            self.advance()
            self.expect_self("(")
            a = self.a_expr()
            args = [a]
            if self.accept_self(","):
                args.append(A.Const(self.advance().value.upper(), "str", pos=t.pos))
            self.expect_self(")")
            return A.FuncCall(["normalize"], args, pos=t.pos, special="normalize")
        if w == "grouping" and self.peek().is_self("("):
            self.advance()
            self.expect_self("(")
            args = [self.a_expr()]
            while self.accept_self(","):
                args.append(self.a_expr())
            self.expect_self(")")
            return A.Grouping(args, pos=t.pos)
        if w == "interval":
            self.advance()
            prec = None
            if self.is_self("("):
                self.advance()
                prec = int(self.advance().value)
                self.expect_self(")")
            st = self.tok
            if st.kind != "STR":
                raise self.error()
            self.advance()
            typ = A.TypeName(["interval"], pos=t.pos)
            fields = self.interval_fields()
            if fields:
                typ.interval_fields = fields
            if prec is not None:
                typ.mods = [A.Const(prec, "int")]
            return A.TypedLiteral(typ, st.value, pos=t.pos)
        if w in _TYPE_KW:
            # a type keyword: typed literal (TIMESTAMP '...', INT '5', DOUBLE PRECISION '1.5')
            save = self.i
            try:
                typ = self.typename(no_array=True)
            except PgError:
                self.i = save
                typ = None
            if typ is not None and self.tok.kind == "STR":
                s = self.advance()
                return A.TypedLiteral(typ, s.value, pos=t.pos)
            self.i = save
            if self._cat(t) == "C":
                if self.peek().is_self("("):
                    raise self.error(self.peek(1) if self.peek(1).kind != "EOF" else None)
                return self.name_primary()
            raise self.error()
        if w == "default":
            self.advance()
            return A.SetToDefault(pos=t.pos)
        if w in ("json_object", "json_array", "json_objectagg", "json_arrayagg", "json_scalar", "json_serialize",
                 "json_query", "json_value", "json_exists", "json_table"):
            raise Unsupported(w.upper() + "()", t.pos + 1)
        if w in ("xmlelement", "xmlforest", "xmlparse", "xmlpi", "xmlroot", "xmlserialize", "xmlconcat",
                 "xmlexists", "xmlattributes", "xmltable"):
            raise Unsupported("XML functions", t.pos + 1)
        return None

    def array_literal(self, pos):
        self.expect_self("[")
        items = []
        if not self.is_self("]"):
            while True:
                if self.is_self("["):
                    items.append(self.array_literal(self.tok.pos))
                else:
                    items.append(self.a_expr())
                if not self.accept_self(","):
                    break
        self.expect_self("]")
        return A.ArrayExpr(items, pos=pos)

    def substring_expr(self):
        t = self.advance()
        self.expect_self("(")
        a = self.a_expr()
        if self.accept_self(","):
            args = [a, self.a_expr()]
            if self.accept_self(","):
                args.append(self.a_expr())
            self.expect_self(")")
            return A.FuncCall(["substring"], args, pos=t.pos)
        if self.accept_kw("from"):
            b = self.a_expr()
            if self.accept_kw("for"):
                c = self.a_expr()
                self.expect_self(")")
                return A.FuncCall(["substring"], [a, b, c], pos=t.pos, special="substring_from")
            self.expect_self(")")
            return A.FuncCall(["substring"], [a, b], pos=t.pos, special="substring_from")
        if self.accept_kw("for"):
            c = self.a_expr()
            if self.accept_kw("from"):
                b = self.a_expr()
                self.expect_self(")")
                return A.FuncCall(["substring"], [a, b, c], pos=t.pos, special="substring_from")
            self.expect_self(")")
            return A.FuncCall(["substring"], [a, A.Const(1, "int"), c], pos=t.pos, special="substring_for")
        if self.accept_kw("similar"):
            b = self.a_expr()
            self.expect_kw("escape")
            c = self.a_expr()
            self.expect_self(")")
            return A.FuncCall(["substring"], [a, b, c], pos=t.pos, special="substring_similar")
        self.expect_self(")")
        return A.FuncCall(["substring"], [a], pos=t.pos)

    def trim_expr(self):
        t = self.advance()
        self.expect_self("(")
        where = "both"
        if self.is_kw("both", "leading", "trailing"):
            where = self.advance().value
        chars = None
        if self.accept_kw("from"):
            args = [self.a_expr()]
            while self.accept_self(","):
                args.append(self.a_expr())
        else:
            first = self.a_expr()
            if self.accept_kw("from"):
                chars = first
                args = [self.a_expr()]
                while self.accept_self(","):
                    args.append(self.a_expr())
            else:
                args = [first]
                while self.accept_self(","):
                    args.append(self.a_expr())
        self.expect_self(")")
        name = {"both": "btrim", "leading": "ltrim", "trailing": "rtrim"}[where]
        if chars is not None:
            args = args[:1] + [chars]
        return A.FuncCall([name], args, pos=t.pos, special="trim")

    def case_expr(self):
        t = self.expect_kw("case")
        arg = None
        if not self.is_kw("when"):
            arg = self.a_expr()
        whens = []
        while self.accept_kw("when"):
            c = self.a_expr()
            self.expect_kw("then")
            r = self.a_expr()
            whens.append((c, r))
        if not whens:
            raise self.error()
        default = None
        if self.accept_kw("else"):
            default = self.a_expr()
        self.expect_kw("end")
        return A.Case(arg, whens, default, pos=t.pos)

    def name_primary(self):
        """A column reference, a function call, or a generic typed literal (jsonb '{}')."""
        t = self.tok
        pos = t.pos
        names = [self.advance().value]
        star = False
        while self.is_self("."):
            nt = self.peek()
            if nt.is_self("*"):
                self.advance()
                self.advance()
                star = True
                break
            if nt.kind in ("IDENT", "QIDENT", "KW"):
                self.advance()
                names.append(self.advance().value)
            else:
                raise self.error(nt)
        if self.is_self("(") and not star:
            if t.kind == "KW" and self._cat(t) == "C" and len(names) == 1:
                raise self.error(self.tok)
            return self.func_call(names, pos)
        if self.tok.kind == "STR" and not star and (len(names) > 1 or t.kind != "KW" or self._cat(t) != "C"):
            s = self.advance()
            return A.TypedLiteral(A.TypeName(names, pos=pos), s.value, pos=pos)
        if t.kind == "KW" and self._cat(t) == "T" and len(names) == 1:
            raise self.error(t)
        e = A.ColumnRef(names + (["*"] if star else []), pos=pos)
        while self.is_self("["):
            e = self.subscript(e)
        return e

    def func_call(self, names, pos):
        self.expect_self("(")
        fc = A.FuncCall(names, [], pos=pos)
        if self.is_self("*"):
            self.advance()
            fc.star = True
            self.expect_self(")")
        elif self.accept_self(")"):
            pass
        else:
            if self.accept_kw("distinct"):
                fc.distinct = True
            else:
                self.accept_kw("all")
            while True:
                if self.accept_kw("variadic"):
                    fc.variadic = True
                aname = None
                if self.is_type_func_name() or self.is_colid():
                    nt = self.peek()
                    if nt.kind == "OP" and nt.value in ("=>", ":="):
                        aname = self.advance().value
                        self.advance()
                fc.args.append(self.a_expr())
                fc.argnames.append(aname)
                if not self.accept_self(","):
                    break
            if self.is_kw("order"):
                fc.order_by = self.sort_clause()
            self.expect_self(")")
        if not any(fc.argnames):
            fc.argnames = []
        if self.is_kw("within") and self.peek().is_kw("group"):
            self.advance()
            self.advance()
            self.expect_self("(")
            fc.order_by = self.sort_clause()
            fc.within_group = True
            self.expect_self(")")
        if self.is_kw("filter") and self.peek().is_self("("):
            self.advance()
            self.advance()
            self.expect_kw("where")
            fc.filter = self.a_expr()
            self.expect_self(")")
        if self.accept_kw("over"):
            if self.is_self("("):
                fc.over = self.window_spec()
            else:
                wp = self.tok.pos
                fc.over = A.WindowDef(ref=self.colid(), pos=wp, bare_ref=True)
        return fc

    # ------------------------------------------------------------ type names
    def typename(self, no_array: bool = False) -> A.TypeName:
        pos = self.tok.pos
        setof = bool(self.accept_kw("setof"))
        typ = self.simple_typename()
        typ.pos = pos
        typ.setof = setof
        if no_array:
            return typ
        if self.is_kw("array"):
            self.advance()
            typ.array = 1
            if self.accept_self("["):
                if self.tok.kind == "INT":
                    self.advance()
                self.expect_self("]")
            return typ
        while self.is_self("["):
            self.advance()
            if self.tok.kind == "INT":
                self.advance()
            self.expect_self("]")
            typ.array += 1
        return typ

    def opt_paren_ints(self) -> list:
        mods = []
        if self.is_self("("):
            self.advance()
            while True:
                t = self.tok
                if t.kind == "INT":
                    mods.append(A.Const(int(self.advance().value), "int", pos=t.pos))
                elif t.is_self("-") and self.peek().kind == "INT":
                    self.advance()
                    mods.append(A.Const(-int(self.advance().value), "int", pos=t.pos))
                elif t.kind in ("STR", "NUM", "IDENT"):
                    mods.append(A.Const(self.advance().value, "str", pos=t.pos))
                else:
                    raise self.error()
                if not self.accept_self(","):
                    break
            self.expect_self(")")
        return mods

    def simple_typename(self) -> A.TypeName:
        t = self.tok
        w = t.value if t.kind == "KW" else None
        pos = t.pos
        if w in ("int", "integer"):
            self.advance()
            return A.TypeName(["int4"], pos=pos)
        if w == "smallint":
            self.advance()
            return A.TypeName(["int2"], pos=pos)
        if w == "bigint":
            self.advance()
            return A.TypeName(["int8"], pos=pos)
        if w == "real":
            self.advance()
            return A.TypeName(["float4"], pos=pos)
        if w == "float":
            self.advance()
            mods = self.opt_paren_ints()
            if mods:
                p = mods[0].value
                if p < 1:
                    raise PgError("precision for type float must be at least 1 bit", "22023", pos=pos + 1)
                if p > 53:
                    raise PgError("precision for type float must be less than 54 bits", "22023", pos=pos + 1)
                return A.TypeName(["float4" if p <= 24 else "float8"], pos=pos)
            return A.TypeName(["float8"], pos=pos)
        if w == "double":
            self.advance()
            self.expect_kw("precision")
            return A.TypeName(["float8"], pos=pos)
        if w in ("decimal", "dec", "numeric"):
            self.advance()
            return A.TypeName(["numeric"], self.opt_paren_ints(), pos=pos)
        if w == "boolean":
            self.advance()
            return A.TypeName(["bool"], pos=pos)
        if w in ("char", "character", "nchar", "national"):
            self.advance()
            if w == "national":
                self.expect_kw("char", "character")
            varying = bool(self.accept_kw("varying"))
            mods = self.opt_paren_ints()
            if varying:
                return A.TypeName(["varchar"], mods, pos=pos)
            return A.TypeName(["bpchar"], mods or [A.Const(1, "int")], pos=pos)
        if w == "varchar":
            self.advance()
            return A.TypeName(["varchar"], self.opt_paren_ints(), pos=pos)
        if w == "bit":
            self.advance()
            varying = bool(self.accept_kw("varying"))
            mods = self.opt_paren_ints()
            if varying:
                return A.TypeName(["varbit"], mods, pos=pos)
            return A.TypeName(["bit"], mods or [A.Const(1, "int")], pos=pos)
        if w in ("timestamp", "time"):
            self.advance()
            mods = self.opt_paren_ints()
            tz = None
            if self.is_kw("with") and self.peek().is_kw("time"):
                self.advance()
                self.advance()
                self.expect_kw("zone")
                tz = True
            elif self.is_kw("without") and self.peek().is_kw("time"):
                self.advance()
                self.advance()
                self.expect_kw("zone")
                tz = False
            name = w + ("tz" if tz else "")
            return A.TypeName([name], mods, pos=pos, tz=tz)
        if w == "interval":
            self.advance()
            typ = A.TypeName(["interval"], pos=pos)
            mods = self.opt_paren_ints()
            f = self.interval_fields()
            if f:
                typ.interval_fields = f
            if mods:
                typ.mods = mods
            return typ
        if self.is_type_func_name():
            names = [self.advance().value]
            while self.is_self("."):
                self.advance()
                names.append(self.col_label())
            mods = self.opt_paren_ints() if self.is_self("(") else []
            return A.TypeName(names, mods, pos=pos)
        raise self.error()

    def interval_fields(self) -> str | None:
        if not self.is_kw("year", "month", "day", "hour", "minute", "second"):
            return None
        a = self.advance().value
        if a == "second" and self.is_self("("):
            self.opt_paren_ints()
        if self.accept_kw("to"):
            b = self.expect_kw("year", "month", "day", "hour", "minute", "second").value
            if b == "second" and self.is_self("("):
                self.opt_paren_ints()
            return f"{a} to {b}"
        return a

    # ------------------------------------------------------------ INSERT / UPDATE / DELETE / MERGE
    def insert_stmt(self) -> A.Insert:
        pos = self.expect_kw("insert").pos
        self.expect_kw("into")
        rv = self.qualified_name()
        if self.accept_kw("as"):
            rv.alias = A.Alias(self.colid(), pos=self.toks[self.i - 1].pos)
        cols = []
        if self.is_self("(") and not self._starts_select(self.i + 1):
            self.advance()
            while True:
                cp = self.tok.pos
                name = self.colid()
                ind = []
                while self.is_self(".") or self.is_self("["):
                    if self.accept_self("."):
                        ind.append(self.col_label())
                    else:
                        self.advance()
                        ind.append(self.a_expr())
                        self.expect_self("]")
                cols.append((name, cp) if not ind else (name, cp, ind))
                if not self.accept_self(","):
                    break
            self.expect_self(")")
        overriding = None
        if self.accept_kw("overriding"):
            overriding = self.expect_kw("system", "user").value
            self.expect_kw("value")
        if self.is_kw("default") and self.peek().is_kw("values"):
            self.advance()
            self.advance()
            source = None
        else:
            source = self.select_stmt()
        ins = A.Insert(rv, cols, source, overriding=overriding, pos=pos)
        if self.is_kw("on") and self.peek().is_kw("conflict"):
            ins.on_conflict = self.on_conflict()
        if self.accept_kw("returning"):
            ins.returning = self.target_list()
        return ins

    def on_conflict(self) -> A.OnConflict:
        pos = self.advance().pos
        self.advance()
        cols, cname, where = [], None, None
        if self.is_self("("):
            self.advance()
            while True:
                ep = self.tok.pos
                if self.is_self("("):
                    self.advance()
                    cols.append(self.a_expr())
                    self.expect_self(")")
                else:
                    cols.append(self.a_expr())
                self.accept_kw("asc", "desc")
                if not self.accept_self(","):
                    break
            self.expect_self(")")
            if self.accept_kw("where"):
                where = self.a_expr()
        elif self.accept_kw("on"):
            self.expect_kw("constraint")
            cname = self.colid()
        self.expect_kw("do")
        if self.accept_kw("nothing"):
            return A.OnConflict(cols, cname, where, "nothing", pos=pos)
        self.expect_kw("update")
        self.expect_kw("set")
        sets = self.set_clause_list()
        uw = None
        if self.accept_kw("where"):
            uw = self.a_expr()
        return A.OnConflict(cols, cname, where, "update", sets, uw, pos=pos)

    def set_clause_list(self) -> list:
        out = [self.set_clause()]
        while self.accept_self(","):
            out.append(self.set_clause())
        return out

    def set_target(self):
        cp = self.tok.pos
        name = self.colid()
        ind = []
        while self.is_self(".") or self.is_self("["):
            if self.accept_self("."):
                ind.append(self.col_label())
            else:
                self.advance()
                ind.append(self.a_expr())
                self.expect_self("]")
        return (name, ind, cp)

    def set_clause(self) -> A.SetClause:
        pos = self.tok.pos
        if self.is_self("("):
            self.advance()
            cols = [self.set_target()]
            while self.accept_self(","):
                cols.append(self.set_target())
            self.expect_self(")")
            self.expect_self("=")
            if self.is_kw("row"):
                v = self.primary()
            elif self.is_self("(") and self._starts_select(self.i + 1):
                self.advance()
                q = self.select_stmt()
                self.expect_self(")")
                v = A.SubLink("multiexpr", q, pos=pos)
            else:
                self.expect_self("(")
                items = [self.expr_or_default()]
                while self.accept_self(","):
                    items.append(self.expr_or_default())
                self.expect_self(")")
                v = A.RowExpr(items, pos=pos)
            return A.SetClause(cols, v, True, pos=pos)
        tgt = self.set_target()
        self.expect_self("=")
        return A.SetClause([tgt], self.expr_or_default(), False, pos=pos)

    def update_stmt(self) -> A.Update:
        pos = self.expect_kw("update").pos
        rv = self.relation_expr()
        rv.alias = self.opt_alias(allow_set=False)
        if rv.alias is not None and rv.alias.cols:
            raise self.error()
        self.expect_kw("set")
        sets = self.set_clause_list()
        frm = self.from_list() if self.accept_kw("from") else []
        where = None
        if self.accept_kw("where"):
            if self.is_kw("current"):
                raise Unsupported("WHERE CURRENT OF", self.tok.pos + 1)
            where = self.a_expr()
        up = A.Update(rv, sets, frm, where, pos=pos)
        if self.accept_kw("returning"):
            up.returning = self.target_list()
        return up

    def delete_stmt(self) -> A.Delete:
        pos = self.expect_kw("delete").pos
        self.expect_kw("from")
        rv = self.relation_expr()
        rv.alias = self.opt_alias()
        using = self.from_list() if self.accept_kw("using") else []
        where = None
        if self.accept_kw("where"):
            where = self.a_expr()
        d = A.Delete(rv, using, where, pos=pos)
        if self.accept_kw("returning"):
            d.returning = self.target_list()
        return d

    def merge_stmt(self) -> A.Merge:
        pos = self.expect_kw("merge").pos
        self.expect_kw("into")
        tgt = self.relation_expr()
        tgt.alias = self.opt_alias()
        self.expect_kw("using")
        src = self.table_primary()
        self.expect_kw("on")
        on = self.a_expr()
        whens = []
        while self.is_kw("when"):
            wp = self.advance().pos
            matched = True
            if self.accept_kw("not"):
                matched = False
            self.expect_kw("matched")
            cond = None
            if self.accept_kw("and"):
                cond = self.a_expr()
            self.expect_kw("then")
            mw = A.MergeWhen(matched, cond, "", pos=wp)
            if self.accept_kw("do"):
                self.expect_kw("nothing")
                mw.action = "nothing"
            elif matched and self.accept_kw("update"):
                self.expect_kw("set")
                mw.action = "update"
                mw.sets = self.set_clause_list()
            elif matched and self.accept_kw("delete"):
                mw.action = "delete"
            elif not matched and self.accept_kw("insert"):
                mw.action = "insert"
                if self.is_self("("):
                    mw.cols = [(c, 0) for c in self.paren_name_list()]
                if self.accept_kw("overriding"):
                    self.expect_kw("system", "user")
                    self.expect_kw("value")
                if self.is_kw("default") and self.peek().is_kw("values"):
                    self.advance()
                    self.advance()
                    mw.values = None
                else:
                    self.expect_kw("values")
                    mw.values = self.values_row()
            else:
                raise self.error()
            whens.append(mw)
        if not whens:
            raise self.error()
        return A.Merge(tgt, src, on, whens, pos=pos)

    # ------------------------------------------------------------ CREATE
    def create_stmt(self):
        pos = self.expect_kw("create").pos
        replace = False
        if self.is_kw("or") and self.peek().is_kw("replace"):
            self.advance()
            self.advance()
            replace = True
        temp = False
        unlogged = False
        if self.is_kw("local", "global") and self.peek().is_kw("temp", "temporary"):
            self.advance()
        if self.accept_kw("temp", "temporary"):
            temp = True
        elif self.accept_kw("unlogged"):
            unlogged = True
        t = self.tok
        if t.is_kw("table"):
            return self.create_table(pos, temp, unlogged)
        if t.is_kw("unique") or t.is_kw("index"):
            return self.create_index(pos)
        if t.is_kw("view") or t.is_kw("recursive"):
            return self.create_view(pos, replace, temp, False)
        if t.is_kw("materialized"):
            self.advance()
            return self.create_view(pos, replace, temp, True)
        if t.is_kw("schema"):
            self.advance()
            ine = self.accept_kws("if", "not", "exists")
            name = None
            if not self.is_kw("authorization"):
                name = self.colid()
            auth = None
            if self.accept_kw("authorization"):
                auth = self.colid()
            return A.CreateSchema(name or auth, ine, auth, pos=pos)
        if t.is_kw("sequence"):
            self.advance()
            ine = self.accept_kws("if", "not", "exists")
            rv = self.qualified_name()
            opts = self.seq_options()
            return A.CreateSequence(rv, opts, ine, temp, pos=pos)
        if t.is_kw("database"):
            self.advance()
            name = self.colid()
            opts = []
            while self.tok.kind != "EOF" and not self.is_self(";"):
                opts.append(self.advance().text)
            return A.CreateDatabase(name, opts, pos=pos)
        if t.is_kw("type"):
            self.advance()
            rv = self.qualified_name()
            self.expect_kw("as")
            if not self.accept_kw("enum"):
                raise Unsupported("CREATE TYPE (other than AS ENUM)", self.tok.pos + 1)
            self.expect_self("(")
            labels = []
            if self.tok.kind == "STR":
                labels.append(self.advance().value)
                while self.accept_self(","):
                    if self.tok.kind != "STR":
                        raise self.error()
                    labels.append(self.advance().value)
            self.expect_self(")")
            return A.CreateType(rv, labels, pos=pos)
        if t.is_kw("extension"):
            self.advance()
            ine = self.accept_kws("if", "not", "exists")
            name = self.colid()
            while self.tok.kind != "EOF" and not self.is_self(";"):
                self.advance()
            return A.CreateExtension(name, ine, pos=pos)
        if t.is_kw("role", "user", "group"):
            self.advance()
            while self.tok.kind != "EOF" and not self.is_self(";"):
                self.advance()
            return A.Noop("CREATE ROLE", pos=pos)
        if t.is_kw("function", "procedure", "trigger", "domain", "aggregate", "operator", "rule", "policy",
                   "publication", "subscription", "server", "tablespace", "event", "foreign", "language", "cast",
                   "collation", "conversion", "statistics", "transform", "access", "text"):
            raise Unsupported("CREATE " + t.text.upper(), t.pos + 1)
        raise self.error()

    def seq_options(self) -> list:
        opts = []
        while True:
            t = self.tok
            if t.is_kw("as"):
                self.advance()
                opts.append(("as", self.typename()))
            elif t.is_kw("increment"):
                self.advance()
                self.accept_kw("by")
                opts.append(("increment", self.signed_int()))
            elif t.is_kw("minvalue"):
                self.advance()
                opts.append(("minvalue", self.signed_int()))
            elif t.is_kw("maxvalue"):
                self.advance()
                opts.append(("maxvalue", self.signed_int()))
            elif t.is_kw("no") and self.peek().is_kw("minvalue", "maxvalue", "cycle"):
                self.advance()
                opts.append(("no " + self.advance().value, None))
            elif t.is_kw("start"):
                self.advance()
                self.accept_kw("with")
                opts.append(("start", self.signed_int()))
            elif t.is_kw("restart"):
                self.advance()
                if self.accept_kw("with") or self.tok.kind == "INT" or self.is_self("-"):
                    opts.append(("restart", self.signed_int()))
                else:
                    opts.append(("restart", None))
            elif t.is_kw("cache"):
                self.advance()
                opts.append(("cache", self.signed_int()))
            elif t.is_kw("cycle"):
                self.advance()
                opts.append(("cycle", True))
            elif t.is_kw("owned"):
                self.advance()
                self.expect_kw("by")
                opts.append(("owned", self.any_name()))
            elif t.is_kw("sequence") and self.peek().is_kw("name"):
                self.advance()
                self.advance()
                opts.append(("name", self.any_name()))
            else:
                return opts

    def signed_int(self) -> int:
        neg = False
        if self.accept_self("-"):
            neg = True
        elif self.accept_self("+"):
            pass
        t = self.tok
        if t.kind not in ("INT", "NUM"):
            raise self.error()
        self.advance()
        v = int(float(t.value)) if t.kind == "NUM" else int(t.value)
        return -v if neg else v

    def create_table(self, pos, temp, unlogged):
        self.expect_kw("table")
        ine = self.accept_kws("if", "not", "exists")
        rv = self.qualified_name()
        ct = A.CreateTable(rv, [], ine, temp, unlogged, pos=pos)
        if self.is_self("(") and not self._paren_then_as():
            self.advance()
            if not self.is_self(")"):
                while True:
                    ct.elements.append(self.table_element())
                    if not self.accept_self(","):
                        break
            self.expect_self(")")
            if self.accept_kw("inherits"):
                raise Unsupported("INHERITS", self.toks[self.i - 1].pos + 1)
            if self.is_kw("partition"):
                raise Unsupported("partitioned tables", self.tok.pos + 1)
            self.table_tail(ct)
            return ct
        # CREATE TABLE name [(cols)] AS query [WITH [NO] DATA]
        if self.is_self("("):
            ct.as_cols = self.paren_name_list()
        self.table_tail(ct)
        self.expect_kw("as")
        if self.is_kw("execute"):
            raise Unsupported("CREATE TABLE AS EXECUTE", self.tok.pos + 1)
        ct.as_query = self.select_stmt()
        if self.accept_kw("with"):
            if self.accept_kw("no"):
                ct.with_data = False
            self.expect_kw("data")
        return ct

    def _paren_then_as(self) -> bool:
        """CREATE TABLE t (a, b) AS ...: a column-name list, not definitions."""
        depth = 0
        j = self.i
        while True:
            t = self.toks[j]
            if t.kind == "EOF":
                return False
            if t.is_self("("):
                depth += 1
            elif t.is_self(")"):
                depth -= 1
                if depth == 0:
                    return self.toks[j + 1].is_kw("as")
            j += 1

    def table_tail(self, ct):
        if self.is_kw("with") and self.peek().is_self("("):
            self.advance()
            self.advance()
            depth = 1
            while depth and self.tok.kind != "EOF":
                if self.is_self("("):
                    depth += 1
                elif self.is_self(")"):
                    depth -= 1
                self.advance()
        if self.accept_kw("on"):
            self.expect_kw("commit")
            if self.accept_kws("preserve", "rows"):
                ct.on_commit = "preserve"
            elif self.accept_kws("delete", "rows"):
                ct.on_commit = "delete"
            else:
                self.expect_kw("drop")
                ct.on_commit = "drop"
        if self.accept_kw("tablespace"):
            self.colid()

    def table_element(self):
        t = self.tok
        if t.is_kw("constraint", "check", "unique", "primary", "foreign", "exclude"):
            return self.table_constraint()
        if t.is_kw("like"):
            self.advance()
            rv = self.qualified_name()
            opts = []
            while self.is_kw("including", "excluding"):
                opts.append((self.advance().value, self.advance().value))
            return ("like", rv, opts)
        pos = t.pos
        name = self.colid()
        typ = self.typename()
        cd = A.ColumnDef(name, typ, [], pos=pos)
        while True:
            c = self.column_constraint()
            if c is None:
                break
            cd.constraints.append(c)
        return cd

    def column_constraint(self):
        pos = self.tok.pos
        name = None
        if self.accept_kw("constraint"):
            name = self.colid()
        t = self.tok
        if t.is_kw("collate"):
            self.advance()
            self.any_name()
            return A.Constraint("collate", name, pos=pos)
        if t.is_kw("not") and self.peek().is_kw("null"):
            self.advance()
            self.advance()
            return A.Constraint("notnull", name, pos=pos)
        if t.is_kw("null"):
            self.advance()
            return A.Constraint("null", name, pos=pos)
        if t.is_kw("default"):
            self.advance()
            start = self.tok.pos
            e = self.b_expr()
            return A.Constraint("default", name, expr=e, pos=pos, expr_text=self.sql[start:self.last_end()])
        if t.is_kw("check"):
            self.advance()
            self.expect_self("(")
            start = self.tok.pos
            e = self.a_expr()
            text = self.sql[start:self.last_end()]
            self.expect_self(")")
            c = A.Constraint("check", name, expr=e, pos=pos, expr_text=text)
            if self.accept_kw("no"):
                self.expect_kw("inherit")
            self.constraint_attrs(c)
            return c
        if t.is_kw("unique"):
            self.advance()
            c = A.Constraint("unique", name, pos=pos)
            if self.accept_kw("nulls"):
                c.nulls_not_distinct = bool(self.accept_kw("not"))
                self.expect_kw("distinct")
            self.index_params()
            self.constraint_attrs(c)
            return c
        if t.is_kw("primary"):
            self.advance()
            self.expect_kw("key")
            c = A.Constraint("primary", name, pos=pos)
            self.index_params()
            self.constraint_attrs(c)
            return c
        if t.is_kw("references"):
            self.advance()
            c = A.Constraint("foreign", name, pos=pos)
            c.ref_table = self.qualified_name()
            if self.is_self("("):
                c.ref_cols = self.paren_name_list()
            self.fk_actions(c)
            self.constraint_attrs(c)
            return c
        if t.is_kw("generated"):
            self.advance()
            if self.accept_kw("always"):
                kind = "always"
            else:
                self.expect_kw("by")
                self.expect_kw("default")
                kind = "by default"
            self.expect_kw("as")
            if self.accept_kw("identity"):
                c = A.Constraint("identity", name, identity=kind, pos=pos)
                if self.accept_self("("):
                    c.seq_opts = self.seq_options()
                    self.expect_self(")")
                return c
            self.expect_self("(")
            start = self.tok.pos
            e = self.a_expr()
            text = self.sql[start:self.last_end()]
            self.expect_self(")")
            self.expect_kw("stored")
            return A.Constraint("generated", name, expr=e, pos=pos, expr_text=text)
        if t.is_kw("deferrable", "initially") or (t.is_kw("not") and self.peek().is_kw("deferrable")):
            c = A.Constraint("attr", name, pos=pos)
            self.constraint_attrs(c)
            return c
        if name is not None:
            raise self.error()
        return None

    def index_params(self):
        if self.is_kw("include"):
            self.advance()
            self.paren_name_list()
        if self.is_kw("with") and self.peek().is_self("("):
            self.advance()
            self.advance()
            while not self.is_self(")") and self.tok.kind != "EOF":
                self.advance()
            self.expect_self(")")
        if self.is_kw("using") and self.peek().is_kw("index"):
            self.advance()
            self.advance()
            self.expect_kw("tablespace")
            self.colid()

    def constraint_attrs(self, c):
        while True:
            if self.accept_kw("deferrable"):
                c.deferrable = True
            elif self.is_kw("not") and self.peek().is_kw("deferrable"):
                self.advance()
                self.advance()
            elif self.accept_kw("initially"):
                self.expect_kw("deferred", "immediate")
            elif self.is_kw("not") and self.peek().is_kw("valid"):
                self.advance()
                self.advance()
                c.not_valid = True
            else:
                return

    def fk_actions(self, c):
        if self.accept_kw("match"):
            self.expect_kw("full", "partial", "simple")
        while self.is_kw("on"):
            self.advance()
            which = self.expect_kw("delete", "update").value
            if self.accept_kw("cascade"):
                act = "c"
            elif self.accept_kw("restrict"):
                act = "r"
            elif self.accept_kws("no", "action"):
                act = "a"
            elif self.accept_kw("set"):
                act = "n" if self.expect_kw("null", "default").value == "null" else "d"
                if self.is_self("("):
                    self.paren_name_list()
            else:
                raise self.error()
            if which == "delete":
                c.on_delete = act
            else:
                c.on_update = act

    def table_constraint(self) -> A.Constraint:
        pos = self.tok.pos
        name = None
        if self.accept_kw("constraint"):
            name = self.colid()
        t = self.tok
        if t.is_kw("check"):
            self.advance()
            self.expect_self("(")
            start = self.tok.pos
            e = self.a_expr()
            text = self.sql[start:self.last_end()]
            self.expect_self(")")
            c = A.Constraint("check", name, expr=e, pos=pos, expr_text=text)
            if self.accept_kw("no"):
                self.expect_kw("inherit")
            self.constraint_attrs(c)
            return c
        if t.is_kw("unique"):
            self.advance()
            c = A.Constraint("unique", name, pos=pos)
            if self.accept_kw("nulls"):
                c.nulls_not_distinct = bool(self.accept_kw("not"))
                self.expect_kw("distinct")
            c.cols = self.paren_name_list()
            self.index_params()
            self.constraint_attrs(c)
            return c
        if t.is_kw("primary"):
            self.advance()
            self.expect_kw("key")
            c = A.Constraint("primary", name, pos=pos)
            c.cols = self.paren_name_list()
            self.index_params()
            self.constraint_attrs(c)
            return c
        if t.is_kw("foreign"):
            self.advance()
            self.expect_kw("key")
            c = A.Constraint("foreign", name, pos=pos)
            c.cols = self.paren_name_list()
            self.expect_kw("references")
            c.ref_table = self.qualified_name()
            if self.is_self("("):
                c.ref_cols = self.paren_name_list()
            self.fk_actions(c)
            self.constraint_attrs(c)
            return c
        if t.is_kw("exclude"):
            raise Unsupported("EXCLUDE constraints", t.pos + 1)
        raise self.error()

    def create_index(self, pos):
        unique = bool(self.accept_kw("unique"))
        self.expect_kw("index")
        conc = bool(self.accept_kw("concurrently"))
        ine = self.accept_kws("if", "not", "exists")
        name = None
        if not self.is_kw("on"):
            name = self.colid()
        self.expect_kw("on")
        rv = self.relation_expr()
        ci = A.CreateIndex(name, rv, [], unique, if_not_exists=ine, concurrently=conc, pos=pos)
        if self.accept_kw("using"):
            ci.method = self.colid()
        self.expect_self("(")
        while True:
            ci.elems.append(self.index_elem())
            if not self.accept_self(","):
                break
        self.expect_self(")")
        if self.accept_kw("include"):
            ci.include = self.paren_name_list()
        if self.accept_kw("nulls"):
            ci.nulls_not_distinct = bool(self.accept_kw("not"))
            self.expect_kw("distinct")
        if self.is_kw("with") and self.peek().is_self("("):
            self.advance()
            self.advance()
            while not self.is_self(")") and self.tok.kind != "EOF":
                self.advance()
            self.expect_self(")")
        if self.accept_kw("tablespace"):
            self.colid()
        if self.accept_kw("where"):
            start = self.tok.pos
            ci.where = self.a_expr()
            ci.where_text = self.sql[start:self.last_end()]
        return ci

    def index_elem(self) -> A.IndexElem:
        pos = self.tok.pos
        if self.is_self("("):
            self.advance()
            start = self.tok.pos
            e = self.a_expr()
            text = self.sql[start:self.last_end()]
            self.expect_self(")")
            el = A.IndexElem(e, None, pos=pos, expr_text=text)
        elif self._is_func_start():
            start = self.tok.pos
            e = self.primary()
            el = A.IndexElem(e, None, pos=pos, expr_text=self.sql[start:self.last_end()])
        else:
            name = self.colid()
            el = A.IndexElem(None, name, pos=pos)
        if self.accept_kw("collate"):
            self.any_name()
        if self.is_colid() and not self.is_kw("asc", "desc", "nulls"):
            el.opclass = self.any_name()
        if self.accept_kw("asc"):
            pass
        elif self.accept_kw("desc"):
            el.desc = True
        if self.accept_kw("nulls"):
            el.nulls = self.expect_kw("first", "last").value
        return el

    def create_view(self, pos, replace, temp, materialized):
        recursive = bool(self.accept_kw("recursive"))
        if recursive:
            raise Unsupported("CREATE RECURSIVE VIEW", pos + 1)
        self.expect_kw("view")
        ine = self.accept_kws("if", "not", "exists") if materialized else False
        rv = self.qualified_name()
        cols = self.paren_name_list() if self.is_self("(") else []
        if self.is_kw("with") and self.peek().is_self("("):
            self.advance()
            self.advance()
            while not self.is_self(")") and self.tok.kind != "EOF":
                self.advance()
            self.expect_self(")")
        self.expect_kw("as")
        start = self.tok.pos
        q = self.select_stmt()
        text = self.sql[start:self.last_end()]
        cv = A.CreateView(rv, cols, q, text, replace, materialized, temp=temp, if_not_exists=ine, pos=pos)
        if materialized:
            if self.accept_kw("with"):
                if self.accept_kw("no"):
                    cv.with_data = False
                self.expect_kw("data")
        elif self.accept_kw("with"):
            if self.is_kw("cascaded", "local"):
                cv.check_option = self.advance().value
            else:
                cv.check_option = "cascaded"
            self.expect_kw("check")
            self.expect_kw("option")
        return cv

    def refresh_stmt(self):
        pos = self.expect_kw("refresh").pos
        self.expect_kw("materialized")
        self.expect_kw("view")
        conc = bool(self.accept_kw("concurrently"))
        rv = self.qualified_name()
        r = A.RefreshMatView(rv, conc, pos=pos)
        if self.accept_kw("with"):
            if self.accept_kw("no"):
                r.with_data = False
            self.expect_kw("data")
        return r

    # ------------------------------------------------------------ ALTER / DROP / TRUNCATE
    def alter_stmt(self):
        pos = self.expect_kw("alter").pos
        t = self.tok
        if t.is_kw("table"):
            self.advance()
            ie = self.accept_kws("if", "exists")
            rv = self.relation_expr()
            at = A.AlterTable(rv, [], ie, pos=pos)
            if self.is_kw("rename"):
                at.cmds.append(self.alter_rename())
                return at
            if self.is_kw("set") and self.peek().is_kw("schema"):
                self.advance()
                self.advance()
                at.cmds.append(A.AlterCmd("set_schema", newname=self.colid(), pos=self.toks[self.i - 1].pos))
                return at
            while True:
                at.cmds.append(self.alter_cmd())
                if not self.accept_self(","):
                    break
            return at
        if t.is_kw("view", "index", "sequence", "schema", "materialized", "database", "type"):
            kind = self.advance().value
            if kind == "materialized":
                self.expect_kw("view")
                kind = "materialized view"
            ie = self.accept_kws("if", "exists")
            rv = self.qualified_name()
            al = A.AlterSimple(kind, rv, "", if_exists=ie, pos=pos)
            if self.accept_kw("rename"):
                if self.accept_kw("to"):
                    al.action = "rename"
                    al.newname = self.colid()
                    return al
                self.accept_kw("column")
                old = self.colid()
                self.expect_kw("to")
                al.action = "rename_column"
                al.options = [old]
                al.newname = self.colid()
                return al
            if kind == "sequence":
                al.action = "options"
                al.options = self.seq_options()
                return al
            if self.accept_kws("owner", "to"):
                self.colid()
                al.action = "owner"
                return al
            if kind == "type" and self.accept_kw("add"):
                self.expect_kw("value")
                ine = self.accept_kws("if", "not", "exists")
                label = self.advance().value
                al.action = "add_value"
                al.options = [label]
                if self.is_kw("before", "after"):
                    al.options.append((self.advance().value, self.advance().value))
                return al
            if self.accept_kw("set"):
                if self.accept_kw("schema"):
                    al.action = "set_schema"
                    al.newname = self.colid()
                    return al
            raise Unsupported(f"this ALTER {kind.upper()} form", pos + 1)
        if t.is_kw("role", "user", "default", "system", "function", "extension"):
            while self.tok.kind != "EOF" and not self.is_self(";"):
                self.advance()
            return A.Noop("ALTER " + t.text.upper(), pos=pos)
        raise self.error()

    def alter_rename(self):
        pos = self.expect_kw("rename").pos
        if self.accept_kw("to"):
            return A.AlterCmd("rename_table", newname=self.colid(), pos=pos)
        if self.accept_kw("constraint"):
            old = self.colid()
            self.expect_kw("to")
            return A.AlterCmd("rename_constraint", name=old, newname=self.colid(), pos=pos)
        self.accept_kw("column")
        old = self.colid()
        self.expect_kw("to")
        return A.AlterCmd("rename_column", name=old, newname=self.colid(), pos=pos)

    def alter_cmd(self) -> A.AlterCmd:
        t = self.tok
        pos = t.pos
        if t.is_kw("add"):
            self.advance()
            if self.is_kw("constraint", "check", "unique", "primary", "foreign", "exclude"):
                c = self.table_constraint()
                return A.AlterCmd("add_constraint", constraint=c, pos=pos)
            self.accept_kw("column")
            ine = self.accept_kws("if", "not", "exists")
            el = self.table_element()
            if not isinstance(el, A.ColumnDef):
                raise self.error()
            return A.AlterCmd("add_column", coldef=el, if_not_exists=ine, pos=pos)
        if t.is_kw("drop"):
            self.advance()
            if self.accept_kw("constraint"):
                ie = self.accept_kws("if", "exists")
                name = self.colid()
                casc = self.drop_behavior()
                return A.AlterCmd("drop_constraint", name=name, if_exists=ie, cascade=casc, pos=pos)
            self.accept_kw("column")
            ie = self.accept_kws("if", "exists")
            name = self.colid()
            casc = self.drop_behavior()
            return A.AlterCmd("drop_column", name=name, if_exists=ie, cascade=casc, pos=pos)
        if t.is_kw("alter"):
            self.advance()
            self.accept_kw("column")
            name = self.colid()
            if self.accept_kw("type") or self.accept_kws("set", "data", "type"):
                typ = self.typename()
                if self.accept_kw("collate"):
                    self.any_name()
                using = None
                if self.accept_kw("using"):
                    using = self.a_expr()
                return A.AlterCmd("alter_type", name=name, typ=typ, using=using, pos=pos)
            if self.accept_kws("set", "default"):
                return A.AlterCmd("set_default", name=name, expr=self.a_expr(), pos=pos)
            if self.accept_kws("drop", "default"):
                return A.AlterCmd("drop_default", name=name, pos=pos)
            if self.accept_kws("set", "not", "null"):
                return A.AlterCmd("set_not_null", name=name, pos=pos)
            if self.accept_kws("drop", "not", "null"):
                return A.AlterCmd("drop_not_null", name=name, pos=pos)
            if self.accept_kw("add"):
                c = self.column_constraint()
                if c is None or c.kind != "identity":
                    raise self.error()
                return A.AlterCmd("add_identity", name=name, constraint=c, pos=pos)
            if self.accept_kws("drop", "identity"):
                ie = self.accept_kws("if", "exists")
                return A.AlterCmd("drop_identity", name=name, if_exists=ie, pos=pos)
            if self.accept_kws("set", "statistics"):
                self.signed_int()
                return A.AlterCmd("noop", name=name, pos=pos)
            raise self.error()
        if t.is_kw("validate"):
            self.advance()
            self.expect_kw("constraint")
            return A.AlterCmd("validate_constraint", name=self.colid(), pos=pos)
        if t.is_kw("owner"):
            self.advance()
            self.expect_kw("to")
            self.colid()
            return A.AlterCmd("noop", pos=pos)
        if t.is_kw("enable", "disable", "replica", "cluster", "set", "reset", "inherit", "no", "attach", "detach"):
            raise Unsupported(f"ALTER TABLE ... {t.text.upper()}", pos + 1)
        raise self.error()

    def drop_behavior(self) -> bool:
        if self.accept_kw("cascade"):
            return True
        self.accept_kw("restrict")
        return False

    def drop_stmt(self):
        pos = self.expect_kw("drop").pos
        t = self.tok
        if t.is_kw("materialized"):
            self.advance()
            self.expect_kw("view")
            kind = "materialized view"
        elif t.is_kw("table", "view", "index", "sequence", "schema", "type", "database", "extension", "function",
                     "role", "user", "domain", "procedure", "trigger"):
            kind = self.advance().value
        else:
            raise self.error()
        conc = False
        if kind == "index" and self.accept_kw("concurrently"):
            conc = True
        ie = self.accept_kws("if", "exists")
        names = []
        if kind in ("role", "user"):
            while self.tok.kind != "EOF" and not self.is_self(";"):
                self.advance()
            return A.Noop("DROP ROLE", pos=pos)
        while True:
            if kind in ("schema", "database", "extension"):
                np = self.tok.pos
                names.append(A.RangeVar(None, self.colid(), pos=np))
            else:
                names.append(self.qualified_name())
                if kind in ("function", "procedure") and self.is_self("("):
                    depth = 0
                    while True:
                        if self.is_self("("):
                            depth += 1
                        elif self.is_self(")"):
                            depth -= 1
                        self.advance()
                        if depth == 0:
                            break
            if not self.accept_self(","):
                break
        casc = self.drop_behavior()
        return A.Drop(kind, names, ie, casc, pos=pos, concurrently=conc)

    def truncate_stmt(self):
        pos = self.expect_kw("truncate").pos
        self.accept_kw("table")
        tables = [self.relation_expr()]
        while self.accept_self(","):
            tables.append(self.relation_expr())
        tr = A.Truncate(tables, pos=pos)
        if self.accept_kws("restart", "identity"):
            tr.restart_identity = True
        elif self.accept_kws("continue", "identity"):
            pass
        tr.cascade = self.drop_behavior()
        return tr

    # ------------------------------------------------------------ transactions, utility
    def txn_stmt(self):
        t = self.advance()
        w = t.value
        pos = t.pos
        if w in ("begin", "start"):
            if w == "start":
                self.expect_kw("transaction")
            else:
                self.accept_kw("work", "transaction")
            opts = self.txn_modes()
            return A.Transaction("begin", options=opts, pos=pos, word=w)
        if w in ("commit", "end"):
            self.accept_kw("work", "transaction")
            if self.accept_kw("prepared"):
                raise Unsupported("COMMIT PREPARED", pos + 1)
            chain = self.opt_chain()
            return A.Transaction("commit", chain=chain, pos=pos, word=w)
        if w in ("rollback", "abort"):
            self.accept_kw("work", "transaction")
            if w == "rollback" and self.accept_kw("to"):
                self.accept_kw("savepoint")
                return A.Transaction("rollback_to", self.colid(), pos=pos, word=w)
            if self.accept_kw("prepared"):
                raise Unsupported("ROLLBACK PREPARED", pos + 1)
            chain = self.opt_chain()
            return A.Transaction("rollback", chain=chain, pos=pos, word=w)
        if w == "savepoint":
            return A.Transaction("savepoint", self.colid(), pos=pos, word=w)
        if w == "release":
            self.accept_kw("savepoint")
            return A.Transaction("release", self.colid(), pos=pos, word=w)
        raise self.error(t)

    def opt_chain(self) -> bool:
        if self.accept_kw("and"):
            no = bool(self.accept_kw("no"))
            self.expect_kw("chain")
            return not no
        return False

    def txn_modes(self) -> list:
        opts = []
        while True:
            if self.accept_kws("isolation", "level"):
                if self.accept_kw("serializable"):
                    opts.append("serializable")
                elif self.accept_kws("repeatable", "read"):
                    opts.append("repeatable read")
                elif self.accept_kws("read", "committed"):
                    opts.append("read committed")
                elif self.accept_kws("read", "uncommitted"):
                    opts.append("read uncommitted")
                else:
                    raise self.error()
            elif self.accept_kws("read", "only"):
                opts.append("read only")
            elif self.accept_kws("read", "write"):
                opts.append("read write")
            elif self.accept_kw("deferrable"):
                opts.append("deferrable")
            elif self.accept_kws("not", "deferrable"):
                opts.append("not deferrable")
            else:
                return opts
            self.accept_self(",")

    def explain_stmt(self):
        pos = self.expect_kw("explain").pos
        opts = {}
        if self.is_self("("):
            self.advance()
            while True:
                name = self.col_label()
                val = True
                if not (self.is_self(",") or self.is_self(")")):
                    vt = self.advance()
                    if vt.kind == "KW" and vt.value in ("true", "on"):
                        val = True
                    elif vt.kind == "KW" and vt.value in ("false", "off"):
                        val = False
                    elif vt.kind == "INT":
                        val = vt.value != "0"
                    else:
                        val = vt.value if vt.kind != "KW" else vt.value
                        if isinstance(val, str) and val in ("true", "on", "false", "off"):
                            val = val in ("true", "on")
                if name == "analyse":
                    name = "analyze"
                if name not in ("analyze", "verbose", "costs", "buffers", "timing", "summary", "format", "settings",
                                "wal", "generic_plan", "memory", "serialize"):
                    raise PgError(f'unrecognized EXPLAIN option "{name}"', "42601",
                                  pos=self.toks[self.i - 1].pos + 1)
                opts[name] = val
                if not self.accept_self(","):
                    break
            self.expect_self(")")
        else:
            if self.accept_kw("analyze", "analyse"):
                opts["analyze"] = True
            if self.accept_kw("verbose"):
                opts["verbose"] = True
        t = self.tok
        if t.is_kw("select", "values", "with", "table") or t.is_self("("):
            st = self.with_stmt() if t.is_kw("with") else self.select_stmt()
        elif t.is_kw("insert"):
            st = self.insert_stmt()
        elif t.is_kw("update"):
            st = self.update_stmt()
        elif t.is_kw("delete"):
            st = self.delete_stmt()
        elif t.is_kw("merge"):
            st = self.merge_stmt()
        elif t.is_kw("create"):
            start = self.tok.pos
            st = self.create_stmt()
            if not (isinstance(st, A.CreateTable) and st.as_query is not None) and \
                    not (isinstance(st, A.CreateView) and st.materialized):
                raise self.error(self.toks[[i for i, x in enumerate(self.toks) if x.pos == start][0]])
        elif t.is_kw("execute"):
            st = self.execute_stmt()
        else:
            raise self.error()
        return A.Explain(st, opts, pos=pos)

    def analyze_stmt(self):
        t = self.advance()
        a = A.Analyze([], pos=t.pos, vacuum=t.value == "vacuum")
        if self.is_self("("):
            self.advance()
            while not self.is_self(")") and self.tok.kind != "EOF":
                w = self.advance().value
                if w == "full":
                    a.full = True
                elif w == "verbose":
                    a.verbose = True
                elif w in ("analyze", "analyse"):
                    a.analyze = True
            self.expect_self(")")
        while True:
            if self.accept_kw("full"):
                a.full = True
            elif self.accept_kw("freeze"):
                pass
            elif self.accept_kw("verbose"):
                a.verbose = True
            elif a.vacuum and self.accept_kw("analyze", "analyse"):
                a.analyze = True
            else:
                break
        if self.tok.kind != "EOF" and not self.is_self(";"):
            while True:
                rv = self.qualified_name()
                if self.is_self("("):
                    rv.cols = self.paren_name_list()
                a.tables.append(rv)
                if not self.accept_self(","):
                    break
        return a

    def var_name(self) -> str:
        parts = [self.col_label()]
        while self.accept_self("."):
            parts.append(self.col_label())
        return ".".join(parts)

    def var_value(self) -> str:
        t = self.tok
        if t.kind == "STR":
            self.advance()
            return t.value
        if t.kind in ("INT", "NUM"):
            self.advance()
            return t.value
        if t.is_self("-") or t.is_self("+"):
            self.advance()
            n = self.advance()
            return ("-" if t.value == "-" else "") + n.value
        if t.kind in ("IDENT", "QIDENT", "KW"):
            self.advance()
            return t.value
        raise self.error()

    def set_stmt(self):
        pos = self.expect_kw("set").pos
        local = False
        if self.accept_kw("local"):
            local = True
        else:
            self.accept_kw("session")
            if self.is_kw("characteristics"):
                while self.tok.kind != "EOF" and not self.is_self(";"):
                    self.advance()
                return A.VarSet("characteristics", [], pos=pos)
        if self.is_kw("transaction"):
            self.advance()
            modes = self.txn_modes()
            return A.VarSet("transaction", modes, pos=pos)
        if self.accept_kws("time", "zone"):
            if self.accept_kw("local") or self.accept_kw("default"):
                return A.VarSet("timezone", [], local, pos=pos)
            if self.is_kw("interval"):
                raise Unsupported("SET TIME ZONE INTERVAL", self.tok.pos + 1)
            return A.VarSet("timezone", [self.var_value()], local, pos=pos)
        if self.accept_kw("schema"):
            return A.VarSet("search_path", [self.var_value()], local, pos=pos)
        if self.accept_kw("names"):
            return A.VarSet("client_encoding", [self.var_value()] if not self.is_kw("default") else [], local,
                            pos=pos)
        if self.is_kw("role", "session") and not self.peek().is_self("="):
            while self.tok.kind != "EOF" and not self.is_self(";"):
                self.advance()
            return A.VarSet("role", [], pos=pos)
        name = self.var_name()
        if not (self.accept_kw("to") or self.accept_self("=")):
            raise self.error()
        if self.accept_kw("default"):
            return A.VarSet(name, [], local, pos=pos)
        vals = [self.var_value()]
        while self.accept_self(","):
            vals.append(self.var_value())
        return A.VarSet(name, vals, local, pos=pos)

    def reset_stmt(self):
        pos = self.expect_kw("reset").pos
        if self.accept_kws("time", "zone"):
            return A.VarSet("timezone", [], reset=True, pos=pos)
        if self.accept_kw("all"):
            return A.VarSet("all", [], reset=True, pos=pos)
        return A.VarSet(self.var_name(), [], reset=True, pos=pos)

    def show_stmt(self):
        pos = self.expect_kw("show").pos
        if self.accept_kws("time", "zone"):
            return A.VarShow("timezone", pos=pos)
        if self.accept_kws("transaction", "isolation", "level"):
            return A.VarShow("transaction_isolation", pos=pos)
        if self.accept_kw("all"):
            return A.VarShow("all", pos=pos)
        return A.VarShow(self.var_name(), pos=pos)

    def copy_stmt(self):
        pos = self.expect_kw("copy").pos
        table = None
        query = None
        cols = []
        if self.is_self("("):
            self.advance()
            t = self.tok
            if t.is_kw("insert", "update", "delete"):
                query = {"insert": self.insert_stmt, "update": self.update_stmt, "delete": self.delete_stmt}[t.value]()
            else:
                query = self.select_stmt()
            self.expect_self(")")
        else:
            table = self.qualified_name()
            if self.is_self("("):
                cols = self.paren_name_list()
        if self.accept_kw("to"):
            to = True
        else:
            self.expect_kw("from")
            to = False
        target = None
        if self.accept_kw("stdin", "stdout"):
            target = None
        elif self.accept_kw("program"):
            raise Unsupported("COPY ... PROGRAM", pos + 1)
        elif self.tok.kind == "STR":
            target = self.advance().value
        else:
            raise self.error()
        opts = self.copy_options()
        if self.accept_kw("where"):
            opts["where"] = self.a_expr()
        return A.Copy(table, cols, query, to, target, opts, pos=pos)

    def copy_options(self) -> dict:
        opts = {}
        self.accept_kw("with")
        if self.is_self("("):
            self.advance()
            while True:
                name = self.col_label()
                val = True
                if not (self.is_self(",") or self.is_self(")")):
                    if self.is_self("("):
                        val = self.paren_name_list()
                    elif self.is_self("*"):
                        self.advance()
                        val = "*"
                    else:
                        vt = self.advance()
                        val = vt.value
                        if vt.kind == "KW" and vt.value in ("true", "on"):
                            val = True
                        elif vt.kind == "KW" and vt.value in ("false", "off"):
                            val = False
                if name in ("force_quote", "force_not_null", "force_null") and val is True:
                    val = []
                opts[name] = val
                if not self.accept_self(","):
                    break
            self.expect_self(")")
            return opts
        # the old syntax: BINARY, DELIMITER [AS] 'c', NULL [AS] 'x', CSV, HEADER, QUOTE, ESCAPE, FORCE ...
        while True:
            if self.accept_kw("binary"):
                opts["format"] = "binary"
            elif self.accept_kw("csv"):
                opts["format"] = "csv"
            elif self.accept_kw("header"):
                opts["header"] = True
            elif self.accept_kw("delimiter"):
                self.accept_kw("as")
                opts["delimiter"] = self.advance().value
            elif self.accept_kw("null"):
                self.accept_kw("as")
                opts["null"] = self.advance().value
            elif self.accept_kw("quote"):
                self.accept_kw("as")
                opts["quote"] = self.advance().value
            elif self.accept_kw("escape"):
                self.accept_kw("as")
                opts["escape"] = self.advance().value
            elif self.accept_kw("encoding"):
                opts["encoding"] = self.advance().value
            elif self.accept_kw("force"):
                if self.accept_kw("quote"):
                    if self.accept_self("*"):
                        opts["force_quote"] = "*"
                    else:
                        opts["force_quote"] = self.name_list()
                elif self.accept_kw("not"):
                    self.expect_kw("null")
                    opts["force_not_null"] = self.name_list()
                else:
                    self.expect_kw("null")
                    opts["force_null"] = self.name_list()
            else:
                return opts

    def comment_stmt(self):
        pos = self.expect_kw("comment").pos
        self.expect_kw("on")
        if self.accept_kw("materialized"):
            self.expect_kw("view")
            kind = "materialized view"
        else:
            kind = self.advance().value
        target = self.any_name()
        self.expect_kw("is")
        if self.accept_kw("null"):
            text = None
        else:
            if self.tok.kind != "STR":
                raise self.error()
            text = self.advance().value
        return A.Comment(kind, target, text, pos=pos)

    def prepare_stmt(self):
        pos = self.expect_kw("prepare").pos
        if self.is_kw("transaction"):
            raise Unsupported("PREPARE TRANSACTION", pos + 1)
        name = self.colid()
        types = []
        if self.accept_self("("):
            types.append(self.typename())
            while self.accept_self(","):
                types.append(self.typename())
            self.expect_self(")")
        self.expect_kw("as")
        start = self.tok.pos
        st = self.statement()
        st.text = self.sql[start:self.last_end()]
        return A.Prepare(name, types, st, pos=pos)

    def execute_stmt(self):
        pos = self.expect_kw("execute").pos
        name = self.colid()
        params = []
        if self.accept_self("("):
            params.append(self.a_expr())
            while self.accept_self(","):
                params.append(self.a_expr())
            self.expect_self(")")
        return A.Execute(name, params, pos=pos)

    def deallocate_stmt(self):
        pos = self.expect_kw("deallocate").pos
        self.accept_kw("prepare")
        if self.accept_kw("all"):
            return A.Deallocate(None, pos=pos)
        return A.Deallocate(self.colid(), pos=pos)

    def grant_stmt(self):
        t = self.advance()
        start = t.pos
        while self.tok.kind != "EOF" and not self.is_self(";"):
            self.advance()
        g = A.Grant(self.sql[start:self.last_end()], pos=t.pos)
        g.revoke = t.value == "revoke"
        return g

    # b_expr: a column default (no AND / OR / NOT / IS NULL / LIKE / IN / BETWEEN at the top)
    def b_expr(self):
        left = self.op_expr()
        t = self.tok
        if t.kind in ("OP", "SELF") and t.value in COMPARE:
            self.advance()
            right = self.op_expr()
            return A.Op(t.value, left, right, pos=t.pos)
        if t.is_kw("is") and self.peek().is_kw("distinct") or (t.is_kw("is") and self.peek().is_kw("not") and
                                                               self.peek(2).is_kw("distinct")):
            self.advance()
            neg = bool(self.accept_kw("not"))
            self.expect_kw("distinct")
            self.expect_kw("from")
            return A.Distinct(left, self.op_expr(), neg, pos=t.pos)
        return left


# keywords that can't be a bare column alias right after an expression in a target list
_CLAUSE_STARTERS = ("from", "where", "group", "having", "window", "order", "limit", "offset", "fetch", "for",
                    "into", "union", "intersect", "except", "returning", "on", "using", "as", "and", "or", "not",
                    "is", "isnull", "notnull", "between", "in", "like", "ilike", "similar", "collate", "at",
                    "then", "else", "end", "when")


class _Paren:
    """Parentheses around an expression only group it; the node itself is returned (marked for deparsing)."""

    @staticmethod
    def wrap(e):
        try:
            e.parens = True
        except AttributeError:
            pass
        return e
