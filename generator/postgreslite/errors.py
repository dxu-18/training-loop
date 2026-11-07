"""PostgreSQL errors and notices, as the server reports them (SQLSTATE, message, DETAIL, HINT, position)."""

from __future__ import annotations


class PgError(Exception):
    """An ERROR (or FATAL): aborts the statement, and the transaction block if one is open.

    pos: 1-based character position in the statement text (psql draws `LINE n:` and a caret from it)."""

    def __init__(self, message: str, code: str = "XX000", *, detail: str | None = None, hint: str | None = None,
                 pos: int | None = None, context: str | None = None, severity: str = "ERROR"):
        super().__init__(message)
        self.message = message
        self.code = code
        self.detail = detail
        self.hint = hint
        self.pos = pos
        self.context = context
        self.severity = severity

    def at(self, pos: int | None) -> "PgError":
        """Attach a position (0-based offset into the statement) unless one is set already."""
        if self.pos is None and pos is not None and pos >= 0:
            self.pos = pos + 1
        return self


class Unsupported(PgError):
    """Valid PostgreSQL that postgres-lite doesn't implement: said plainly instead of a wrong answer."""

    def __init__(self, what: str, pos: int | None = None):
        super().__init__(f"{what} is not supported by postgres-lite (the simulator behind this terminal)",
                         "0A000", pos=pos)


def syntax_error(near: str | None, pos: int | None) -> PgError:
    if near is None:
        return PgError("syntax error at end of input", "42601", pos=pos)
    return PgError(f'syntax error at or near "{near}"', "42601", pos=pos)


# a few SQLSTATEs by name (errcodes.txt)
UNDEFINED_COLUMN = "42703"
UNDEFINED_TABLE = "42P01"
UNDEFINED_FUNCTION = "42883"
AMBIGUOUS_COLUMN = "42702"
GROUPING_ERROR = "42803"
DATATYPE_MISMATCH = "42804"
INVALID_TEXT = "22P02"
DIVISION_BY_ZERO = "22012"
NUMERIC_OUT_OF_RANGE = "22003"
UNIQUE_VIOLATION = "23505"
NOT_NULL_VIOLATION = "23502"
CHECK_VIOLATION = "23514"
FK_VIOLATION = "23503"
DUPLICATE_TABLE = "42P07"
DUPLICATE_COLUMN = "42701"
DUPLICATE_OBJECT = "42710"
IN_FAILED_TXN = "25P02"
WINDOWING_ERROR = "42P20"
CARDINALITY = "21000"
STRING_TOO_LONG = "22001"
DATETIME_FIELD_OVERFLOW = "22008"
INVALID_DATETIME = "22007"
FEATURE_NOT_SUPPORTED = "0A000"
DEPENDENT_OBJECTS = "2BP01"
WRONG_OBJECT_TYPE = "42809"
INVALID_PARAMETER = "22023"
SYNTAX_ERROR = "42601"
UNDEFINED_OBJECT = "42704"
INVALID_COLUMN_REFERENCE = "42P10"
CANNOT_COERCE = "42846"
INVALID_JSON = "22P02"
