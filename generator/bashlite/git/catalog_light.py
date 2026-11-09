"""The 30 small practice repositories (1-5 MB each). Every recipe tells a short, plausible project history: who
committed what, the branches and tags, merged and open pull requests, and the issues. Problems lean on these
details (a renamed file for `log --follow`, a regression for `bisect`, a stale branch, a conflicting pair of
branches, a leaked secret in old history...)."""

from __future__ import annotations

import json

from .catalog import LFS, Lazy, recipe
from .catalog_gen import (GITIGNORES, ci_workflow, c_file, csv_text, go_file, ident, jpg, js_module, license_text,
                          md_doc, paragraph, pdf, pick, png, py_function, py_module, py_test, rust_file, sentence)


# ---------------------------------------------------------------------------- shared helpers

def grow(b, n, people, paths, prefix="", msgs=None):
    """n ordinary maintenance commits: a function or a paragraph added to one of `paths` at a time."""
    for _ in range(n):
        p = pick(b.rng, paths)
        who = pick(b.rng, people)
        old = b.read(p)
        base = p.rsplit("/", 1)[-1]
        if p.endswith(".md"):
            new = old.rstrip("\n") + "\n\n" + paragraph(b.rng, None, 2) + "\n"
            msg = pick(b.rng, [f"Update {base}", f"Clarify {base}", f"docs: expand {base}", f"Fix wording in {base}"])
        elif p.endswith(".py"):
            fn = py_function(b.rng)
            name = fn.split("(", 1)[0][4:]
            new = old.rstrip("\n") + "\n\n\n" + fn
            msg = pick(b.rng, [f"Add {name}() to {base}", f"Add {name} helper", f"Move {name} into {base}",
                               f"Implement {name}"])
        elif p.endswith((".js", ".ts", ".jsx")):
            new = old.rstrip("\n") + "\n\n" + js_module(b.rng, sentence(b.rng), 1).split("\n", 2)[2]
            msg = pick(b.rng, [f"Add helper to {base}", f"Refactor {base}", f"Handle empty input in {base}"])
        elif p.endswith(".go"):
            body = go_file(b.rng, "x", "", 1).split(")\n\n", 1)[1]
            new = old.rstrip("\n") + "\n\n" + body
            msg = pick(b.rng, [f"Add helper to {base}", f"Validate input in {base}"])
        elif p.endswith(".rs"):
            new = old.rstrip("\n") + "\n\n" + rust_file(b.rng, "", 1).split("\n\n", 2)[2]
            msg = pick(b.rng, [f"Add counting helper to {base}", f"Refactor {base}"])
        elif p.endswith((".c", ".h")):
            new = old.rstrip("\n") + "\n\n" + c_file(b.rng, "", 1).split("\n\n", 1)[1]
            msg = pick(b.rng, [f"{base}: add line counter", f"{base}: handle NULL input"])
        else:
            new = old.rstrip("\n") + "\n" + sentence(b.rng) + "\n"
            msg = f"Update {base}"
        if msgs:
            msg = pick(b.rng, msgs).format(f=base)
        b.commit(prefix + msg, {p: new}, by=who)


def readme(title, desc, sections):
    out = [f"# {title}", "", desc, ""]
    for h, body in sections:
        out += [f"## {h}", "", body.strip("\n"), ""]
    return "\n".join(out)


def sub(text, old, new):
    assert old in text, old
    return text.replace(old, new, 1)


# ============================================================================ 1. minilibs/requests-mini

API_PY = '''"""The top-level helpers: requests_mini.get(), post() and friends."""

from .sessions import Session


def request(method, url, **kwargs):
    """Send a request with a one-off Session and return the Response."""
    with Session() as session:
        return session.request(method, url, **kwargs)


def get(url, params=None, **kwargs):
    """Send a GET request."""
    return request("get", url, params=params, **kwargs)


def post(url, data=None, json=None, **kwargs):
    """Send a POST request."""
    return request("post", url, data=data, json=json, **kwargs)
'''

SESSIONS_PY = '''"""Session: keeps headers and cookies across requests."""

from .models import Response
from .compat import urlencode


class Session:
    def __init__(self):
        self.headers = {"User-Agent": "requests-mini"}
        self.cookies = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.cookies.clear()

    def request(self, method, url, params=None, data=None, json=None, headers=None):
        if params:
            url = url + "?" + urlencode({k: v for k, v in params.items() if v is not None})
        merged = dict(self.headers, **(headers or {}))
        return self.send(method.upper(), url, merged, data if json is None else json)

    def send(self, method, url, headers, body):
        raise NotImplementedError("transport adapters are not wired up yet")
'''

MODELS_PY = '''"""Response objects."""

import json as _json

from .exceptions import HTTPError


class Response:
    def __init__(self, status_code=200, headers=None, content=b"", url=""):
        self.status_code = status_code
        self.headers = headers or {}
        self.content = content
        self.url = url

    @property
    def ok(self):
        return self.status_code < 400

    @property
    def text(self):
        return self.content.decode(self.encoding or "utf-8", "replace")

    @property
    def encoding(self):
        ctype = self.headers.get("Content-Type", "")
        if "charset=" in ctype:
            return ctype.split("charset=", 1)[1]
        return "utf-8"

    def json(self):
        return _json.loads(self.text)

    def raise_for_status(self):
        if not self.ok:
            raise HTTPError(f"{self.status_code} error for url: {self.url}", response=self)
'''

EXC_PY = '''"""Exceptions raised by requests-mini."""


class RequestException(IOError):
    """Something went wrong while handling a request."""

    def __init__(self, *args, response=None):
        super().__init__(*args)
        self.response = response


class HTTPError(RequestException):
    """The server answered with an error status."""


class ConnectionError(RequestException):
    """The connection failed."""


class Timeout(RequestException):
    """The request timed out."""
'''


@recipe("minilibs", "requests-mini", "A small, readable HTTP client library modelled on requests",
        language="Python", topics=["python", "http", "http-client", "library"], stars=2140,
        homepage="https://minilibs.github.io/requests-mini",
        about="A Python HTTP library with a long tagged history (v0.1.0 to v1.1.0), releases, a renamed module "
              "(compat.py -> _compat.py), merged and open pull requests, and labelled issues.")
def _requests_mini(b):
    core = ["mira-chen", "tomas-ruiz", "priya-n"]
    b.commit("Initial commit", {
        "README.md": readme("requests-mini", "HTTP for people, small enough to read in an afternoon.", [
            ("Install", "```\npip install requests-mini\n```"),
            ("Usage", "```python\nimport requests_mini as requests\nr = requests.get(\"https://example.com\")\n"
                      "print(r.status_code)\n```")]),
        "LICENSE": license_text("Apache-2.0", 2024, "Mira Chen"),
        ".gitignore": GITIGNORES["python"]}, by="mira-chen")
    b.commit("Add Session and the get/post helpers", {
        "requests_mini/__init__.py": '"""requests-mini: HTTP for people."""\n\nfrom .api import get, post, request\n'
                                     '\n__version__ = "0.1.0"\n',
        "requests_mini/api.py": API_PY, "requests_mini/sessions.py": SESSIONS_PY,
        "requests_mini/compat.py": '"""Python version shims."""\n\nfrom urllib.parse import urlencode, urlparse\n'
                                   '\n__all__ = ["urlencode", "urlparse"]\n'}, by="mira-chen")
    b.commit("Add Response model with json() and raise_for_status()", {"requests_mini/models.py": MODELS_PY},
             by="mira-chen")
    b.commit("Add the exception hierarchy", {"requests_mini/exceptions.py": EXC_PY}, by="priya-n")
    b.commit("Add tests for the API helpers", {
        "tests/test_api.py": py_test(b.rng, "requests_mini", ["get", "post", "request"]),
        "requirements-dev.txt": "pytest==7.4.0\nresponses==0.24.1\n",
        ".github/workflows/ci.yml": ci_workflow("python")}, by="tomas-ruiz")
    b.commit("Add pyproject.toml", {"pyproject.toml": '[project]\nname = "requests-mini"\nversion = "0.1.0"\n'
                                                      'requires-python = ">=3.9"\n'}, by="mira-chen")
    b.tag("v0.1.0", "First public release")
    b.issue(1, "get() sends params whose value is None", "dalvarez", "Calling get(url, params={'a': None}) adds "
            "`a=None` to the query string.", labels=["bug"], closed=True)
    b.branch("feature/timeouts")
    b.edit("requests_mini/sessions.py", lambda t: sub(sub(t, "json=None, headers=None):",
                                                          "json=None, headers=None, timeout=None):"),
                                                      "        return self.send(method.upper(), url, merged, data if json "
                                                      "is None else json)",
                                                      "        return self.send(method.upper(), url, merged, data if json "
                                                      "is None else json,\n                         timeout=timeout)"),
           "Add a timeout parameter to Session.request", by="jberg")
    b.edit("README.md", lambda t: t + "\n## Timeouts\n\nPass `timeout=` (seconds) to any helper. Without it a request "
                                      "can wait forever.\n", "Document timeouts", by="jberg")
    b.switch("main")
    b.merge_pr(3, "feature/timeouts", "Add timeouts", by="mira-chen", author="jberg",
               reviews=[("mira-chen", "APPROVED", "Nice, thanks!")])
    b.commit("Rename compat.py to _compat.py", rename={"requests_mini/compat.py": "requests_mini/_compat.py"},
             by="priya-n")
    b.edit("requests_mini/sessions.py", lambda t: sub(t, "from .compat import", "from ._compat import"),
           "Import from the renamed _compat module", by="priya-n")
    b.edit("requests_mini/models.py", lambda t: sub(t, '        return self.content.decode(self.encoding or "utf-8", '
                                                       '"replace")',
                                                    '        if not self.content:\n            return ""\n'
                                                    '        return self.content.decode(self.encoding or "utf-8", '
                                                    '"replace")'),
           "Fix encoding detection for empty bodies", by="tomas-ruiz")
    b.issue(2, "Support HTTP/2?", "kwame-a", "Would HTTP/2 be in scope for a library this size?",
            labels=["enhancement", "question"])
    b.commit("Bump pytest from 7.4.0 to 8.0.0", {"requirements-dev.txt": "pytest==8.0.0\nresponses==0.24.1\n"},
             by="dependabot[bot]")
    b.edit("requests_mini/__init__.py", lambda t: sub(t, '"0.1.0"', '"0.2.0"'), "Bump version to 0.2.0",
           by="mira-chen")
    b.tag("v0.2.0", "Timeouts and encoding fixes")
    b.issue(5, "raise_for_status() message has no reason phrase", "elif-k", labels=["bug"], closed=True)
    b.branch("feature/put-delete")
    b.edit("requests_mini/api.py", lambda t: t + '\n\ndef put(url, data=None, **kwargs):\n    """Send a PUT request."""\n'
                                             '    return request("put", url, data=data, **kwargs)\n',
           "Add put()", by="aiko-t")
    b.edit("requests_mini/api.py", lambda t: t + '\n\ndef delete(url, **kwargs):\n    """Send a DELETE request."""\n'
                                             '    return request("delete", url, **kwargs)\n',
           "Add delete()", by="aiko-t")
    b.edit("requests_mini/__init__.py", lambda t: sub(t, "from .api import get, post, request",
                                                      "from .api import delete, get, post, put, request"),
           "Export put and delete", by="aiko-t")
    b.switch("main")
    b.merge_pr(7, "feature/put-delete", "Add put() and delete() helpers", by="tomas-ruiz", author="aiko-t",
               method="squash")
    b.commit("Add images for the docs site", {
        "docs/index.md": md_doc(b.rng, "requests-mini", ["sessions", "responses", "timeouts", "errors"]),
        "docs/img/logo.png": png(412_000), "docs/img/flow.png": png(655_000),
        "docs/img/sessions.png": png(388_000)}, by="priya-n")
    grow(b, 5, core, ["requests_mini/api.py", "requests_mini/models.py", "docs/index.md", "README.md"])
    b.edit("requests_mini/__init__.py", lambda t: sub(t, '"0.2.0"', '"1.0.0"'), "Release 1.0.0", by="mira-chen")
    b.tag("v1.0.0", "requests-mini 1.0.0\n\nThe API is stable from here on.")
    b.release("v1.0.0", "requests-mini 1.0", "First stable release: get/post/put/delete, sessions, timeouts.")
    b.issue(9, "The timeout docs are confusing", "ben-o", "Is the timeout per read or for the whole request?",
            labels=["documentation", "good first issue"])
    b.branch("fix/redirect-loop")
    b.edit("requests_mini/sessions.py", lambda t: t + '\n    max_redirects = 30\n', "Cap redirects at 30",
           by="sokafor")
    b.switch("main")
    b.merge_pr(12, "fix/redirect-loop", "Stop following redirects after 30 hops", by="mira-chen", author="sokafor")
    b.issue(14, "Add type hints", "lena-f", labels=["enhancement"])
    b.edit("requests_mini/api.py", lambda t: sub(t, "def get(url, params=None, **kwargs):",
                                                 "def get(url: str, params: dict | None = None, **kwargs):"),
           "Add type hints to get()", by="lena-f")
    grow(b, 3, core + ["lena-f"], ["requests_mini/models.py", "docs/index.md"])
    b.edit("requests_mini/__init__.py", lambda t: sub(t, '"1.0.0"', '"1.1.0"'), "Bump version to 1.1.0",
           by="mira-chen")
    b.tag("v1.1.0", "requests-mini 1.1.0")
    b.release("v1.1.0", "requests-mini 1.1", "Type hints, redirect cap, docs.")
    b.branch("feature/retries")
    b.commit("Add a Retry adapter", {"requests_mini/adapters.py": py_module(
        b.rng, "Transport adapters with retries.", 3, ["retry_request", "backoff_delay"])}, by="tomas-ruiz")
    b.edit("requests_mini/adapters.py", lambda t: t + "\n\nMAX_RETRIES = 3\n", "Default to 3 retries",
           by="tomas-ruiz")
    b.switch("main")
    b.open_pr(15, "feature/retries", "Add retry adapter", "tomas-ruiz", body="Adds a Retry adapter with "
              "exponential backoff. Closes #2? Not quite, but related.",
              reviews=[("priya-n", "CHANGES_REQUESTED", "Can we make MAX_RETRIES configurable?")])


# ============================================================================ 2. minilibs/flask-mini

APP_PY = '''"""A tiny WSGI microframework: routes, requests and responses."""

import json


class Flask:
    def __init__(self, name):
        self.name = name
        self.routes = {}

    def route(self, path, methods=("GET",)):
        def register(view):
            for m in methods:
                self.routes[(m, path)] = view
            return view
        return register

    def __call__(self, environ, start_response):
        key = (environ["REQUEST_METHOD"], environ.get("PATH_INFO", "/"))
        view = self.routes.get(key)
        if view is None:
            start_response("404 NOT FOUND", [("Content-Type", "text/plain")])
            return [b"Not Found"]
        body = view()
        if isinstance(body, dict):
            body = json.dumps(body)
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [body.encode()]
'''


@recipe("minilibs", "flask-mini", "A teaching-size web microframework in the style of Flask",
        language="Python", topics=["python", "wsgi", "web-framework"], stars=980,
        about="A microframework with a maintenance branch (2.x) next to main, a security fix made on main that "
              "also belongs on 2.x (cherry-pick), a dev branch, and tags 1.0.0, 2.0.0, 2.0.1.")
def _flask_mini(b):
    b.commit("Initial commit", {"README.md": readme("flask-mini", "A microframework you can read in one sitting.", [
        ("Hello world", "```python\nfrom flask_mini import Flask\napp = Flask(__name__)\n\n@app.route(\"/\")\n"
                        "def index():\n    return \"Hello!\"\n```")]),
        "LICENSE": license_text("BSD-3-Clause", 2024, "minilibs"), ".gitignore": GITIGNORES["python"]},
        by="lena-f")
    b.commit("Add the Flask app object and routing", {"flask_mini/__init__.py": "from .app import Flask\n\n"
                                                      "__version__ = \"1.0.0\"\n", "flask_mini/app.py": APP_PY},
             by="lena-f")
    b.commit("Add hello world example", {"examples/hello.py": 'from flask_mini import Flask\n\napp = Flask(__name__)'
                                         '\n\n\n@app.route("/")\ndef index():\n    return "Hello, world!"\n'},
             by="felix-w")
    b.commit("Add tests", {"tests/test_app.py": py_test(b.rng, "flask_mini", ["Flask"]),
                           ".github/workflows/ci.yml": ci_workflow("python")}, by="felix-w")
    b.tag("1.0.0", "flask-mini 1.0.0", by="lena-f")
    grow(b, 3, ["lena-f", "felix-w"], ["flask_mini/app.py", "README.md"])
    b.edit("flask_mini/app.py", lambda t: sub(t, 'def route(self, path, methods=("GET",)):',
                                              'def route(self, path, methods=("GET",), strict_slashes=True):'),
           "Add strict_slashes to route()", by="lena-f")
    b.edit("flask_mini/__init__.py", lambda t: sub(t, '"1.0.0"', '"2.0.0"'), "Release 2.0.0", by="lena-f")
    b.tag("2.0.0", "flask-mini 2.0.0", by="lena-f")
    b.branch("2.x")
    b.switch("main")
    b.commit("Add JSON responses for dict return values", {"docs/json.md": md_doc(b.rng, "JSON responses",
                                                                                  ["dicts", "headers"]),
                                                           "docs/_static/flask-mini.png": png(910_000),
                                                           "docs/_static/request-cycle.png": png(760_000)},
             by="chloe-m")
    b.branch("dev")
    grow(b, 3, ["chloe-m", "felix-w"], ["flask_mini/app.py", "docs/json.md"])
    b.switch("main")
    b.edit("flask_mini/app.py", lambda t: sub(t, 'start_response("404 NOT FOUND", [("Content-Type", "text/plain")])',
                                              'start_response("404 NOT FOUND", [("Content-Type", "text/plain"),\n'
                                              '                                            ("X-Content-Type-Options", '
                                              '"nosniff")])'),
           "Send X-Content-Type-Options: nosniff on 404 pages\n\nWithout it some browsers guess the type of the "
           "error body.", by="omar-s")
    grow(b, 2, ["lena-f", "chloe-m"], ["README.md", "docs/json.md"])
    b.switch("2.x")
    b.commit("Fix typo in README", {"README.md": b.read("README.md").replace("sitting", "sitting.")}, by="lena-f")
    b.edit("flask_mini/__init__.py", lambda t: sub(t, '"2.0.0"', '"2.0.1"'), "Release 2.0.1", by="lena-f")
    b.tag("2.0.1", "flask-mini 2.0.1 (maintenance)", by="lena-f")
    b.switch("main")
    b.issue(4, "Backport the nosniff header to 2.x", "omar-s", "2.x users get the same 404 pages.",
            labels=["security", "backport"])
    b.open_pr(6, "dev", "Dev: JSON responses and docs", "chloe-m", draft=True)


# ============================================================================ 3. minilibs/leftpad

LEFTPAD_JS = """'use strict';

module.exports = leftPad;

function leftPad(str, len, ch) {
  str = String(str);
  var i = -1;
  if (!ch && ch !== 0) ch = ' ';
  len = len - str.length;
  while (++i < len) {
    str = ch + str;
  }
  return str;
}
"""


@recipe("minilibs", "leftpad", "String left-padding, the tiny module half the ecosystem depends on",
        language="JavaScript", topics=["javascript", "npm", "string"], stars=1210,
        about="A one-function npm package with many small semver tags (v1.0.0 to v1.3.0), a CHANGELOG, a test "
              "file and a performance rewrite.")
def _leftpad(b):
    b.commit("Initial commit", {"index.js": LEFTPAD_JS, "package.json": json.dumps(
        {"name": "leftpad-mini", "version": "1.0.0", "main": "index.js", "license": "WTFPL",
         "scripts": {"test": "node test.js"}}, indent=2) + "\n",
        "README.md": "# leftpad\n\nPad the left side of a string.\n\n```js\nleftPad('foo', 5) // '  foo'\n```\n"},
        by="ben-o")
    b.tag("v1.0.0")
    b.commit("Add tests", {"test.js": "const assert = require('assert');\nconst leftPad = require('./');\n\n"
                                      "assert.strictEqual(leftPad('foo', 5), '  foo');\n"
                                      "assert.strictEqual(leftPad('foobar', 6), 'foobar');\n"
                                      "assert.strictEqual(leftPad(1, 2, 0), '01');\nconsole.log('ok');\n"}, by="ben-o")
    b.tag("v1.0.1")
    b.commit("Handle 0 as a pad character", {"CHANGELOG.md": "# Changelog\n\n## 1.0.2\n- `0` works as a pad "
                                                             "character\n"}, by="grace-l")
    b.tag("v1.0.2")
    b.edit("index.js", lambda t: sub(sub(t, "  while (++i < len) {\n    str = ch + str;\n  }\n  return str;",
                                         "  var pad = '';\n  while (++i < len) {\n    pad += ch;\n  }\n"
                                         "  return pad + str;"), "'use strict';", "'use strict';\n\n// O(n) instead of "
                                                                                  "O(n^2) string building"),
           "Build the padding once instead of prepending in a loop", by="grace-l")
    b.edit("CHANGELOG.md", lambda t: sub(t, "# Changelog\n", "# Changelog\n\n## 1.1.0\n- faster for long pads\n"),
           "Changelog for 1.1.0", by="ben-o")
    b.tag("v1.1.0")
    b.commit("Add a benchmark chart", {"docs/benchmark.png": png(1_150_000)}, by="grace-l")
    b.commit("Add TypeScript definitions", {"index.d.ts": "declare function leftPad(str: string | number, len: "
                                                          "number, ch?: string | number): string;\n"
                                                          "export = leftPad;\n"}, by="zara-q")
    b.edit("CHANGELOG.md", lambda t: sub(t, "# Changelog\n", "# Changelog\n\n## 1.2.0\n- TypeScript types\n"),
           "Changelog for 1.2.0", by="ben-o")
    b.tag("v1.2.0")
    b.edit("README.md", lambda t: t + "\nWorks with numbers too: `leftPad(7, 3, 0) // '007'`.\n",
           "Document number input", by="zara-q")
    b.edit("CHANGELOG.md", lambda t: sub(t, "# Changelog\n", "# Changelog\n\n## 1.3.0\n- docs\n"),
           "Changelog for 1.3.0", by="ben-o")
    b.tag("v1.3.0")
    b.issue(3, "Should leftPad('abc', 5, '') pad with nothing?", "kwame-a", labels=["question"])


# ============================================================================ 4. minilibs/datetime-utils

@recipe("minilibs", "datetime-utils", "Small, tree-shakeable date helpers for TypeScript",
        language="TypeScript", topics=["typescript", "dates", "conventional-commits"], stars=640,
        about="A TypeScript library written with Conventional Commits (feat:, fix:, chore(deps):...), a "
              "CHANGELOG generated from them, and semver tags.")
def _datetime_utils(b):
    b.commit("chore: initial commit", {"package.json": json.dumps(
        {"name": "@minilibs/datetime-utils", "version": "0.1.0", "type": "module", "main": "dist/index.js",
         "scripts": {"build": "tsc", "test": "vitest run"}}, indent=2) + "\n",
        "tsconfig.json": '{\n  "compilerOptions": { "target": "ES2020", "module": "ESNext", "strict": true, '
                         '"outDir": "dist" }\n}\n',
        ".gitignore": GITIGNORES["node"], "README.md": "# datetime-utils\n\nDate helpers, one function per file.\n"},
        by="grace-l")
    b.commit("feat: add addDays and startOfDay", {"src/addDays.ts": "export function addDays(d: Date, n: number): "
                                                  "Date {\n  const out = new Date(d);\n  out.setDate(out.getDate() + "
                                                  "n);\n  return out;\n}\n",
                                                  "src/startOfDay.ts": "export function startOfDay(d: Date): Date {\n"
                                                  "  const out = new Date(d);\n  out.setHours(0, 0, 0, 0);\n"
                                                  "  return out;\n}\n",
                                                  "src/index.ts": "export * from './addDays';\nexport * from "
                                                  "'./startOfDay';\n"}, by="grace-l")
    b.commit("test: cover addDays across DST", {"test/addDays.test.ts": "import { addDays } from '../src/addDays';\n"
                                                "test('adds days', () => {\n  expect(addDays(new Date(2024, 0, 1), 1)"
                                                ".getDate()).toBe(2);\n});\n"}, by="ravi-k")
    b.tag("v0.1.0", "v0.1.0")
    b.commit("feat: add formatISO", {"src/formatISO.ts": "export function formatISO(d: Date): string {\n  return "
                                     "d.toISOString().slice(0, 10);\n}\n"}, by="ravi-k")
    b.edit("src/index.ts", lambda t: t + "export * from './formatISO';\n", "feat: export formatISO", by="ravi-k")
    b.edit("src/formatISO.ts", lambda t: sub(t, "  return d.toISOString().slice(0, 10);",
                                             "  const pad = (n: number) => String(n).padStart(2, '0');\n  return "
                                             "`${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;"),
           "fix: formatISO used UTC instead of local time\n\nCloses #4", by="grace-l")
    b.issue(4, "formatISO is off by one day west of UTC", "dalvarez", labels=["bug"], closed=True)
    b.commit("chore(deps): bump typescript from 5.3.3 to 5.4.5", {"package-lock.json": '{\n  "lockfileVersion": 3,\n'
                                                                  '  "packages": {"node_modules/typescript": '
                                                                  '{"version": "5.4.5"}}\n}\n'},
             by="dependabot[bot]")
    b.commit("docs: add CHANGELOG", {"CHANGELOG.md": "# Changelog\n\n## 0.2.0\n### Features\n- formatISO\n"
                                     "### Bug Fixes\n- formatISO used UTC\n\n## 0.1.0\n- addDays, startOfDay\n"},
             by="grace-l")
    b.tag("v0.2.0", "v0.2.0")
    b.commit("feat: add differenceInDays", {"src/differenceInDays.ts": "const DAY = 24 * 60 * 60 * 1000;\n\n"
                                            "export function differenceInDays(a: Date, b: Date): number {\n  return "
                                            "Math.round((a.getTime() - b.getTime()) / DAY);\n}\n"}, by="zara-q")
    b.edit("src/index.ts", lambda t: t + "export * from './differenceInDays';\n", "feat: export differenceInDays",
           by="zara-q")
    b.commit("refactor!: drop the default export\n\nBREAKING CHANGE: import named functions instead.",
             {"src/index.ts": b.read("src/index.ts")}, by="grace-l")
    b.commit("ci: run tests on pull requests", {".github/workflows/ci.yml": ci_workflow("node")}, by="ravi-k")
    b.commit("docs: add the API page", {"docs/api.md": md_doc(b.rng, "API", ["addDays", "formatISO", "differences"]),
                                        "docs/timeline.png": png(1_480_000)}, by="zara-q")
    b.edit("CHANGELOG.md", lambda t: sub(t, "# Changelog\n", "# Changelog\n\n## 1.0.0\n### BREAKING CHANGES\n- no "
                                                             "default export\n### Features\n- differenceInDays\n"),
           "chore(release): 1.0.0", by="grace-l")
    b.tag("v1.0.0", "v1.0.0")
    b.release("v1.0.0", "v1.0.0", "Breaking: named exports only. New: differenceInDays.", by="grace-l")
    b.branch("feat/is-weekend")
    b.commit("feat: add isWeekend", {"src/isWeekend.ts": "export function isWeekend(d: Date): boolean {\n  const day "
                                     "= d.getDay();\n  return day === 0 || day === 6;\n}\n"}, by="kwame-a")
    b.commit("added tests", {"test/isWeekend.test.ts": "test('saturday', () => {});\n"}, by="kwame-a")
    b.switch("main")
    b.open_pr(9, "feat/is-weekend", "Add isWeekend", "kwame-a", body="Second commit message doesn't follow "
              "Conventional Commits yet, will fix.", comments=[("grace-l", "Please reword 'added tests' to "
                                                                           "'test: add isWeekend tests'.")])


# ============================================================================ 5. mini-ml/nanogpt-mini

MODEL_PY = '''"""A small GPT: token + position embeddings, a stack of transformer blocks, a language-model head."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.n_embd), nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(), nn.Linear(4 * cfg.n_embd, cfg.n_embd))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        T = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        logits = self.head(self.ln_f(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
'''

TRAIN_PY = '''"""Train a character-level GPT. Usage: python train.py config/train_shakespeare.py"""

import os
import sys
import time

import torch

from model import GPT, GPTConfig

batch_size = 64
learning_rate = 1e-3
max_iters = 5000
eval_interval = 500
out_dir = "out"
exec(open(sys.argv[1]).read()) if len(sys.argv) > 1 else None

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(GPTConfig()).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
os.makedirs(out_dir, exist_ok=True)
t0 = time.time()
for it in range(max_iters):
    if it % eval_interval == 0:
        torch.save(model.state_dict(), os.path.join(out_dir, "ckpt.pt"))
        print(f"iter {it}: {time.time() - t0:.1f}s")
'''


@recipe("mini-ml", "nanogpt-mini", "The simplest repository for training small GPTs on characters",
        language="Python", topics=["deep-learning", "transformer", "gpt", "pytorch"], stars=5230,
        about="A compact GPT training repo: model.py, train.py, configs, a data-prep script; checkpoints are "
              "gitignored; an open experiment branch (experiment/rope) and tagged milestones.")
def _nanogpt(b):
    b.commit("initial commit: model and training loop", {
        "model.py": MODEL_PY, "train.py": TRAIN_PY, ".gitignore": GITIGNORES["ml"] + "out/\n*.pt\n",
        "README.md": readme("nanogpt-mini", "Train a small GPT on Shakespeare in a few minutes.", [
            ("Quick start", "```\npython data/shakespeare_char/prepare.py\npython train.py config/train_shakespeare.py"
                            "\npython sample.py --out_dir=out\n```")]),
        "LICENSE": license_text("MIT", 2024, "mini-ml")}, by="aiko-t")
    b.commit("add shakespeare data prep", {"data/shakespeare_char/prepare.py": py_module(
        b.rng, "Download tiny-shakespeare and encode it as characters.", 3, ["encode_text", "decode_ids"],
        ("os", "pickle")), "data/shakespeare_char/readme.md": "Run prepare.py to create train.bin and val.bin.\n"},
        by="aiko-t")
    b.commit("add config for shakespeare", {"config/train_shakespeare.py": "out_dir = 'out-shakespeare-char'\n"
                                            "eval_interval = 250\nmax_iters = 5000\nlearning_rate = 1e-3\n"
                                            "batch_size = 64\n"}, by="aiko-t")
    b.commit("add sampling script", {"sample.py": py_module(b.rng, "Sample text from a trained model.", 2,
                                                            ["load_checkpoint", "generate"], ("torch",))},
             by="aiko-t")
    b.tag("v0.1")
    b.commit("add loss curves to the README", {"assets/loss.png": png(820_000), "assets/samples.png": png(640_000)},
             by="yuki-s")
    grow(b, 4, ["aiko-t", "yuki-s"], ["model.py", "train.py", "README.md"])
    b.edit("config/train_shakespeare.py", lambda t: sub(t, "learning_rate = 1e-3", "learning_rate = 6e-4"),
           "lower the learning rate, 1e-3 diverged after 3k iters", by="yuki-s")
    b.commit("add a GPT-2 finetuning config", {"config/finetune_gpt2.py": "init_from = 'gpt2'\nlearning_rate = 3e-5\n"
                                               "max_iters = 2000\nbatch_size = 8\n"}, by="ravi-k")
    b.tag("v0.2")
    b.branch("experiment/rope")
    b.edit("model.py", lambda t: sub(t, "        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)\n",
                                     "        self.pos = None  # rotary embeddings are applied inside attention\n"),
           "try rotary position embeddings", by="yuki-s")
    b.commit("rope: notes on results", {"notes/rope.md": "val loss 1.47 vs 1.49 with learned positions; "
                                        "needs more runs.\n"}, by="yuki-s")
    b.switch("main")
    b.open_pr(4, "experiment/rope", "Rotary position embeddings (experiment)", "yuki-s", draft=True,
              body="Not ready: results are within noise so far.")
    b.issue(2, "CUDA out of memory with batch_size 64 on a 8GB card", "ben-o", labels=["question"],
            comments=[("aiko-t", "Use batch_size = 16 and gradient accumulation.")], closed=True)
    b.issue(5, "Add a Makefile target for data prep", "zara-q", labels=["good first issue"])


# ============================================================================ 6. mini-ml/sentiment-classifier

@recipe("mini-ml", "sentiment-classifier", "Movie-review sentiment with scikit-learn, data versioned with DVC",
        language="Python", topics=["machine-learning", "scikit-learn", "dvc", "nlp"], stars=310,
        about="An sklearn project whose dataset is tracked with DVC (data/reviews.csv.dvc, a DVC remote that "
              "`dvc pull` can fetch from), params.yaml, metrics.json and a training script.")
def _sentiment(b):
    import hashlib as _h
    words_pos = ["great", "moving", "funny", "brilliant", "warm", "sharp", "clever", "lovely"]
    words_neg = ["dull", "slow", "messy", "flat", "boring", "clumsy", "noisy", "weak"]
    rows = []
    for i in range(1500):
        pos = b.rng.random() < 0.5
        ws = words_pos if pos else words_neg
        rows.append(f"{i},\"a {pick(b.rng, ws)} and {pick(b.rng, ws)} film\",{1 if pos else 0}")
    data = ("id,text,label\n" + "\n".join(rows) + "\n").encode()
    md5 = _h.md5(data).hexdigest()
    remote = "s3://mini-ml-data/sentiment"
    store = b.gh.world.__dict__.setdefault("dvc_remotes", {})
    from ..vfs import LazyData
    store.setdefault(remote, {})[md5] = LazyData(src=data)
    b.commit("Initial project layout", {
        "README.md": readme("sentiment-classifier", "Logistic regression on TF-IDF features.", [
            ("Get the data", "```\npip install dvc\ndvc pull\n```"), ("Train", "```\npython src/train.py\n```")]),
        ".gitignore": GITIGNORES["python"], "requirements.txt": "scikit-learn==1.5.2\npandas==2.2.3\ndvc==3.63.0\n",
        "src/train.py": py_module(b.rng, "Train and evaluate the classifier; writes metrics.json.", 3,
                                  ["load_reviews", "train_model", "evaluate"], ("json", "sys")),
        "params.yaml": "train:\n  C: 1.0\n  max_features: 5000\n"}, by="priya-n")
    b.commit("Track the dataset with DVC", {
        ".dvc/config": f"[core]\n    remote = store\n['remote \"store\"']\n    url = {remote}\n",
        ".dvc/.gitignore": "/config.local\n/tmp\n/cache\n", ".dvcignore": "# DVC ignore patterns\n",
        "data/reviews.csv.dvc": f"outs:\n- md5: {md5}\n  size: {len(data)}\n  hash: md5\n  path: reviews.csv\n",
        "data/.gitignore": "/reviews.csv\n"}, by="priya-n")
    b.commit("Add DVC pipeline", {"dvc.yaml": "stages:\n  train:\n    cmd: python src/train.py\n    deps:\n"
                                              "    - data/reviews.csv\n    - src/train.py\n    params:\n    - train\n"
                                              "    metrics:\n    - metrics.json:\n        cache: false\n"},
             by="ravi-k")
    b.commit("Record baseline metrics", {"metrics.json": '{"accuracy": 0.861, "f1": 0.858}\n'}, by="ravi-k")
    b.commit("Add confusion matrix plot", {"reports/confusion.png": png(1_020_000),
                                           "reports/roc.png": png(870_000)}, by="priya-n")
    grow(b, 3, ["priya-n", "ravi-k"], ["src/train.py", "README.md"])
    b.edit("params.yaml", lambda t: sub(t, "C: 1.0", "C: 0.5"), "Tune C to 0.5", by="ravi-k")
    b.commit("Update metrics after tuning C", {"metrics.json": '{"accuracy": 0.874, "f1": 0.871}\n'}, by="ravi-k")
    b.tag("baseline-v1", "Logistic regression baseline, acc 0.874", by="priya-n")
    b.issue(3, "dvc pull fails without credentials?", "chloe-m", labels=["question"])


# ============================================================================ 7. mini-ml/titanic-eda

def _notebook(rng, cells, outputs=True, img_kb=0):
    nb_cells = []
    for i, src in enumerate(cells, 1):
        outs = []
        if outputs:
            outs.append({"name": "stdout", "output_type": "stream", "text": [sentence(rng) + "\n"]})
            if img_kb:
                outs.append({"data": {"image/png": "iVBORw0KGgo" + "A" * (img_kb * 1024),
                                      "text/plain": ["<Figure size 640x480 with 1 Axes>"]},
                             "metadata": {}, "output_type": "display_data"})
        nb_cells.append({"cell_type": "code", "execution_count": i if outputs else None, "metadata": {},
                         "outputs": outs, "source": [src]})
    nb = {"cells": nb_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                         "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(nb, indent=1) + "\n"


@recipe("mini-ml", "titanic-eda", "Exploratory analysis of the Titanic passenger list in Jupyter",
        language="Jupyter Notebook", topics=["data-analysis", "jupyter", "pandas"], stars=88,
        about="Notebooks committed with their outputs (big base64 plots, execution counts) so every re-run "
              "makes a noisy diff: the case for nbstripout.")
def _titanic(b):
    def row(rng, i):
        return [i + 1, rng.randint(0, 1), rng.randint(1, 3), pick(rng, ["male", "female"]), rng.randint(1, 70),
                round(rng.uniform(5, 260), 2)]
    b.commit("Add data and README", {"data/train.csv": csv_text(b.rng, ["PassengerId", "Survived", "Pclass", "Sex",
                                                                        "Age", "Fare"], row, 891),
                                     "README.md": "# titanic-eda\n\nWho survived, and why? Notebooks in "
                                                  "`notebooks/`.\n", "requirements.txt": "pandas\nmatplotlib\n"
                                                                                         "seaborn\njupyter\n"},
             by="elif-k")
    cells = ["import pandas as pd\ndf = pd.read_csv('../data/train.csv')", "df.describe()",
             "df.groupby('Sex').Survived.mean().plot.bar()", "df.Age.hist(bins=30)"]
    b.commit("First look at the data", {"notebooks/01_explore.ipynb": _notebook(b.rng, cells, True, 180)},
             by="elif-k")
    b.commit("Re-run notebook", {"notebooks/01_explore.ipynb": _notebook(b.rng, cells, True, 190)}, by="elif-k")
    b.commit("Add survival by class", {"notebooks/02_class.ipynb": _notebook(
        b.rng, ["df.groupby('Pclass').Survived.mean()", "sns.barplot(data=df, x='Pclass', y='Survived')"], True, 240)},
        by="noor-h")
    b.commit("Re-run everything", {"notebooks/01_explore.ipynb": _notebook(b.rng, cells, True, 185),
                                   "notebooks/02_class.ipynb": _notebook(
                                       b.rng, ["df.groupby('Pclass').Survived.mean()",
                                               "sns.barplot(data=df, x='Pclass', y='Survived')"], True, 250)},
             by="elif-k")
    b.commit("Add figures for the report", {"figures/survival_by_sex.png": png(430_000),
                                            "figures/age_hist.png": png(380_000)}, by="noor-h")


# ============================================================================ 8. toolsmiths/ripgrep-mini

@recipe("toolsmiths", "ripgrep-mini", "A line-oriented recursive search tool in Rust, ripgrep-style",
        language="Rust", topics=["rust", "cli", "search", "grep"], stars=1730,
        about="A Rust CLI with Cargo.toml, modules for argument parsing and searching, integration tests, "
              "semver tags 0.1.0-0.4.0 and a CHANGELOG.")
def _ripgrep(b):
    b.commit("Initial commit", {"Cargo.toml": '[package]\nname = "rg-mini"\nversion = "0.1.0"\nedition = "2021"\n\n'
                                              '[dependencies]\nregex = "1"\nwalkdir = "2"\n',
                                "src/main.rs": "mod args;\nmod search;\n\nfn main() {\n    let opts = args::parse();\n"
                                               "    std::process::exit(search::run(&opts));\n}\n",
                                "src/args.rs": rust_file(b.rng, "Command-line arguments.", 2),
                                "src/search.rs": rust_file(b.rng, "Searching files line by line.", 3),
                                ".gitignore": GITIGNORES["rust"],
                                "README.md": "# rg-mini\n\n`rg-mini PATTERN [PATH]` searches recursively, skipping "
                                             "what .gitignore ignores.\n",
                                "LICENSE": license_text("MIT", 2024, "toolsmiths")}, by="jberg")
    b.tag("0.1.0")
    b.commit("Add integration tests", {"tests/integration.rs": rust_file(b.rng, "End-to-end tests.", 2),
                                       ".github/workflows/ci.yml": ci_workflow("rust")}, by="felix-w")
    grow(b, 4, ["jberg", "felix-w"], ["src/search.rs", "src/args.rs"])
    b.edit("Cargo.toml", lambda t: sub(t, '"0.1.0"', '"0.2.0"'), "Release 0.2.0", by="jberg")
    b.tag("0.2.0")
    b.commit("Add --count and --files-with-matches", {"src/output.rs": rust_file(b.rng, "Output formats.", 2)},
             by="yuki-s")
    b.commit("Add a demo GIF", {"doc/demo.gif": Lazy(2_100_000, b"GIF89a")}, by="jberg")
    b.edit("Cargo.toml", lambda t: sub(t, '"0.2.0"', '"0.3.0"'), "Release 0.3.0", by="jberg")
    b.tag("0.3.0")
    grow(b, 3, ["jberg", "yuki-s", "felix-w"], ["src/search.rs", "src/output.rs"])
    b.edit("Cargo.toml", lambda t: sub(t, '"0.3.0"', '"0.4.0"'), "Release 0.4.0", by="jberg")
    b.tag("0.4.0")
    b.issue(7, "Binary files are searched", "omar-s", labels=["bug"])


# ============================================================================ 9. toolsmiths/hugo-mini

@recipe("toolsmiths", "hugo-mini", "A fast static site generator written in Go",
        language="Go", topics=["go", "static-site-generator", "markdown"], stars=890,
        about="A Go module (go.mod, internal packages, a themes directory, an example site) with tags v0.1.0 to "
              "v0.3.0, a merged PR and an open one.")
def _hugo(b):
    b.commit("Initial commit", {"go.mod": "module github.com/toolsmiths/hugo-mini\n\ngo 1.22\n",
                                "main.go": 'package main\n\nimport (\n\t"fmt"\n\t"os"\n)\n\nfunc main() {\n\tif '
                                           'len(os.Args) < 2 {\n\t\tfmt.Println("usage: hugo-mini build|serve")\n'
                                           '\t\tos.Exit(1)\n\t}\n}\n',
                                "internal/render/render.go": go_file(b.rng, "render", "turns Markdown into HTML.", 3),
                                "internal/config/config.go": go_file(b.rng, "config", "reads config.toml.", 2),
                                ".gitignore": GITIGNORES["go"], "README.md": "# hugo-mini\n\nBuild a site: "
                                                                             "`hugo-mini build`.\n"}, by="dalvarez")
    b.tag("v0.1.0")
    b.commit("Add the basic theme", {"themes/basic/layouts/page.html": "<!doctype html>\n<title>{{ .Title }}"
                                     "</title>\n<main>{{ .Content }}</main>\n",
                                     "themes/basic/static/style.css": "body { max-width: 42rem; margin: auto; }\n",
                                     "themes/basic/static/fonts/inter.woff2": Lazy(310_000, b"wOF2")},
             by="chloe-m")
    b.commit("Add an example site", {"exampleSite/config.toml": 'title = "Example"\ntheme = "basic"\n',
                                     "exampleSite/content/_index.md": "# Welcome\n\nThis is the example site.\n",
                                     "exampleSite/static/hero.jpg": jpg(1_350_000)}, by="chloe-m")
    grow(b, 4, ["dalvarez", "chloe-m"], ["internal/render/render.go", "internal/config/config.go"])
    b.tag("v0.2.0")
    b.branch("feature/live-reload")
    b.commit("Add live reload to serve", {"internal/server/server.go": go_file(b.rng, "server", "serves the site.",
                                                                              2)}, by="kwame-a")
    b.switch("main")
    b.merge_pr(8, "feature/live-reload", "Live reload for hugo-mini serve", by="dalvarez", author="kwame-a")
    b.tag("v0.3.0")
    b.branch("feature/taxonomies")
    b.commit("WIP taxonomies", {"internal/taxonomy/taxonomy.go": go_file(b.rng, "taxonomy", "groups pages by tag.",
                                                                         2)}, by="ana-p")
    b.switch("main")
    b.open_pr(11, "feature/taxonomies", "Tags and categories", "ana-p")


# ============================================================================ 10. toolsmiths/jq-mini

@recipe("toolsmiths", "jq-mini", "A lightweight command-line JSON processor in C",
        language="C", topics=["c", "json", "cli"], stars=760, default_branch="master",
        about="An older C project on a `master` branch: Makefile, lexer/parser/interpreter sources, shell tests, "
              "tags jq-mini-1.0 and jq-mini-1.1, and a long-lived maintenance branch.")
def _jq(b):
    b.commit("Initial import", {"Makefile": "CC ?= cc\nCFLAGS ?= -O2 -Wall\nOBJ = src/main.o src/lexer.o src/parser.o "
                                            "src/jv.o\n\njq-mini: $(OBJ)\n\t$(CC) -o $@ $(OBJ)\n\ntest: jq-mini\n"
                                            "\tsh tests/run.sh\n\nclean:\n\trm -f jq-mini $(OBJ)\n",
                                "src/main.c": c_file(b.rng, "jq-mini: entry point.", 1, "jv.h"),
                                "src/lexer.c": c_file(b.rng, "Tokenizer for filter expressions.", 3),
                                "src/parser.c": c_file(b.rng, "Recursive-descent parser.", 3),
                                "src/jv.c": c_file(b.rng, "JSON values.", 4, "jv.h"),
                                "src/jv.h": "#ifndef JV_H\n#define JV_H\n#include <stddef.h>\nint jv_count(const char "
                                            "*s, size_t n);\n#endif\n",
                                "tests/run.sh": "#!/bin/sh\nset -e\necho '{\"a\":1}' | ./jq-mini .a\n",
                                ".gitignore": GITIGNORES["c"], "COPYING": license_text("MIT", 2015, "toolsmiths")},
             by="kwame-a", hours=(1, 2))
    grow(b, 5, ["kwame-a", "ben-o"], ["src/lexer.c", "src/parser.c", "src/jv.c"])
    b.tag("jq-mini-1.0", "jq-mini 1.0", by="kwame-a")
    b.branch("jq-mini-1.0-maint")
    b.commit("Fix crash on empty input", {"src/main.c": b.read("src/main.c") + "\n/* empty input is not an error */\n"},
             by="ben-o")
    b.switch("master")
    grow(b, 6, ["kwame-a", "ben-o", "felix-w"], ["src/parser.c", "src/jv.c", "src/lexer.c"])
    b.commit("Add the manual", {"docs/manual.md": md_doc(b.rng, "jq-mini manual", ["filters", "paths", "types"]),
                                "docs/manual.pdf": pdf(1_900_000)}, by="kwame-a")
    b.tag("jq-mini-1.1", "jq-mini 1.1", by="kwame-a")
    grow(b, 3, ["felix-w"], ["src/jv.c"])


# ============================================================================ 11. mira-chen/dotfiles

@recipe("mira-chen", "dotfiles", "My shell, vim, git and tmux configuration",
        language="Shell", topics=["dotfiles", "bash", "vim", "tmux"], stars=41,
        about="A personal dotfiles repo: .bashrc, .vimrc, .gitconfig, .tmux.conf, an install.sh that symlinks "
              "them, and a macos branch with Mac-only tweaks.")
def _dotfiles(b):
    b.commit("Initial dotfiles", {
        "bashrc": "# ~/.bashrc\nexport EDITOR=vim\nexport HISTSIZE=50000\nalias ll='ls -alF'\nalias gs='git status'\n",
        "vimrc": "set number\nset relativenumber\nset expandtab shiftwidth=4\nsyntax on\n",
        "gitconfig": "[user]\n\tname = Mira Chen\n\temail = mira.chen@example.org\n[alias]\n\tst = status -sb\n"
                     "\tlg = log --oneline --graph --decorate\n[init]\n\tdefaultBranch = main\n",
        "install.sh": "#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$0\")\"\nfor f in bashrc vimrc "
                      "gitconfig tmux.conf; do\n  ln -sf \"$PWD/$f\" \"$HOME/.$f\"\ndone\necho \"linked\"\n",
        "README.md": "# dotfiles\n\n```\ngit clone git@github.com:mira-chen/dotfiles.git ~/dotfiles && "
                     "~/dotfiles/install.sh\n```\n"}, by="mira-chen")
    b.commit("Add tmux config", {"tmux.conf": "set -g mouse on\nset -g base-index 1\nbind | split-window -h\n"
                                              "bind - split-window -v\n"}, by="mira-chen")
    for msg, path, add in [("vim: show trailing whitespace", "vimrc", "set list listchars=tab:>-,trail:~\n"),
                           ("bash: nicer prompt with git branch", "bashrc",
                            "PS1='\\u@\\h:\\w$(__git_ps1 \" (%s)\")\\$ '\n"),
                           ("git: add rebase.autosquash", "gitconfig", "[rebase]\n\tautosquash = true\n"),
                           ("bash: add mkcd function", "bashrc", "mkcd() { mkdir -p \"$1\" && cd \"$1\"; }\n"),
                           ("tmux: vim-style pane movement", "tmux.conf", "bind h select-pane -L\nbind l select-pane "
                                                                          "-R\n"),
                           ("vim: leader is space", "vimrc", "let mapleader = \" \"\n"),
                           ("git: push.autoSetupRemote", "gitconfig", "[push]\n\tautoSetupRemote = true\n")]:
        b.edit(path, lambda t, add=add: t + add, msg, by="mira-chen", hours=(6, 200))
    b.commit("Add wallpaper", {"wallpapers/mountains.jpg": jpg(2_600_000)}, by="mira-chen")
    b.branch("macos")
    b.edit("bashrc", lambda t: t + "alias ls='ls -G'\nexport BASH_SILENCE_DEPRECATION_WARNING=1\n",
           "macos: BSD ls colors, silence zsh warning", by="mira-chen")
    b.commit("macos: brew bundle", {"Brewfile": 'brew "git"\nbrew "tmux"\nbrew "ripgrep"\n'}, by="mira-chen")
    b.switch("main")
    b.edit("bashrc", lambda t: t + "export PATH=\"$HOME/.local/bin:$PATH\"\n", "bash: ~/.local/bin on PATH",
           by="mira-chen")


# ============================================================================ 12. mira-chen/mira-chen.github.io

@recipe("mira-chen", "mira-chen.github.io", "Personal site, published with GitHub Pages",
        language="HTML", topics=["github-pages", "website", "portfolio"], stars=12,
        homepage="https://mira-chen.github.io",
        about="A user Pages site served from main at https://mira-chen.github.io: index.html, about.html, posts, "
              "CSS and images.")
def _pages_site(b):
    b.commit("Initial site", {"index.html": "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                                            "<title>Mira Chen</title>\n<link rel=\"stylesheet\" href=\"css/style.css\">"
                                            "\n</head>\n<body>\n<h1>Mira Chen</h1>\n<p>I build small, readable "
                                            "tools.</p>\n<a href=\"about.html\">About</a>\n</body>\n</html>\n",
                              "css/style.css": "body { font-family: system-ui; max-width: 40rem; margin: 2rem auto; }"
                                               "\n", "README.md": "Source of https://mira-chen.github.io\n"},
             by="mira-chen")
    b.commit("Add about page", {"about.html": "<!doctype html>\n<title>About</title>\n<h1>About</h1>\n<p>Engineer, "
                                              "climber, occasional writer.</p>\n<img src=\"img/me.jpg\" alt=\"me\">\n",
                                "img/me.jpg": jpg(740_000)}, by="mira-chen")
    for i, title in enumerate(["Why I keep dotfiles in git", "Bisecting a flaky test", "Notes on rebasing"], 1):
        b.commit(f"Post: {title}", {f"posts/{i:02d}.html": f"<!doctype html>\n<title>{title}</title>\n<h1>{title}</h1>"
                                    f"\n<p>{paragraph(b.rng, None, 4)}</p>\n",
                                    f"img/post{i}.png": png(420_000 + i * 90_000)}, by="mira-chen", hours=(24, 400))
    b.edit("index.html", lambda t: sub(t, "<a href=\"about.html\">About</a>",
                                       "<a href=\"about.html\">About</a>\n<ul>\n<li><a href=\"posts/01.html\">Dotfiles"
                                       "</a></li>\n<li><a href=\"posts/02.html\">Bisect</a></li>\n<li><a href=\"posts/"
                                       "03.html\">Rebasing</a></li>\n</ul>"),
           "Link the posts from the home page", by="mira-chen")
    b.repo.pages = {"branch": "main", "path": "/", "built": b.heads["main"]}


# ============================================================================ 13. priya-n/thesis-latex

@recipe("priya-n", "thesis-latex", "My MSc thesis, written in LaTeX",
        language="TeX", topics=["latex", "thesis"], stars=3, license="",
        about="A LaTeX thesis: main.tex, chapters/*.tex, refs.bib, PDF figures, a Makefile; an advisor-comments "
              "branch with suggested edits, and tags draft-1 and submitted.")
def _thesis(b):
    b.commit("Set up thesis skeleton", {
        "main.tex": "\\documentclass[12pt]{report}\n\\usepackage{graphicx}\n\\title{Learning Under Distribution "
                    "Shift}\n\\author{Priya Nair}\n\\begin{document}\n\\maketitle\n\\tableofcontents\n"
                    "\\include{chapters/intro}\n\\include{chapters/method}\n\\include{chapters/results}\n"
                    "\\bibliographystyle{plain}\n\\bibliography{refs}\n\\end{document}\n",
        "chapters/intro.tex": "\\chapter{Introduction}\n" + paragraph(b.rng, None, 5) + "\n",
        "chapters/method.tex": "\\chapter{Method}\n" + paragraph(b.rng, None, 5) + "\n",
        "chapters/results.tex": "\\chapter{Results}\nTODO\n",
        "refs.bib": "@article{shift2019,\n  title={Covariate shift, a survey},\n  author={Doe, J.},\n  year={2019}\n}\n",
        "Makefile": "thesis.pdf: main.tex chapters/*.tex refs.bib\n\tlatexmk -pdf -jobname=thesis main.tex\n",
        ".gitignore": GITIGNORES["latex"] + "thesis.pdf\n"}, by="priya-n")
    for ch in ("intro", "method", "method", "intro"):
        b.edit(f"chapters/{ch}.tex", lambda t: t + "\n" + paragraph(b.rng, None, 4) + "\n",
               f"Write more of the {ch} chapter", by="priya-n", hours=(10, 90))
    b.commit("Add result figures", {"figures/accuracy.pdf": pdf(820_000), "figures/shift.pdf": pdf(640_000),
                                    "chapters/results.tex": "\\chapter{Results}\n\\begin{figure}\n\\includegraphics"
                                                            "[width=\\linewidth]{figures/accuracy.pdf}\n\\caption{"
                                                            "Accuracy under shift.}\n\\end{figure}\n" +
                                                            paragraph(b.rng, None, 3) + "\n"}, by="priya-n")
    b.tag("draft-1", "First full draft sent to advisor", by="priya-n")
    b.branch("advisor-comments")
    b.edit("chapters/intro.tex", lambda t: t.replace("\\chapter{Introduction}\n", "\\chapter{Introduction}\n% NL: "
                                                     "start with the motivating example\n", 1),
           "Comments on the introduction", by="grace-l")
    b.edit("chapters/results.tex", lambda t: t + "% NL: add error bars\n", "Ask for error bars", by="grace-l")
    b.switch("main")
    b.edit("refs.bib", lambda t: t + "@inproceedings{dro2020,\n  title={Robust optimization for shift},\n  author="
                                     "{Roe, A.},\n  year={2020}\n}\n", "Add DRO reference", by="priya-n")
    b.merge("advisor-comments", "Merge advisor comments", by="priya-n")
    b.edit("chapters/results.tex", lambda t: t.replace("% NL: add error bars\n", "Error bars show one standard "
                                                       "deviation over five seeds.\n"), "Add error bars", by="priya-n")
    b.commit("Add appendix figures", {"figures/appendix_grid.pdf": pdf(1_300_000)}, by="priya-n")
    b.tag("submitted", "Version submitted to the examiners", by="priya-n")


# ============================================================================ 14. tomas-ruiz/leetcode-solutions

PROBLEMS = ["two_sum", "add_two_numbers", "longest_substring", "median_two_arrays", "longest_palindrome",
            "zigzag_conversion", "reverse_integer", "atoi", "palindrome_number", "regex_matching", "container_water",
            "integer_to_roman", "roman_to_integer", "longest_common_prefix", "three_sum", "three_sum_closest",
            "letter_combinations", "four_sum", "remove_nth_node", "valid_parentheses", "merge_two_lists",
            "generate_parentheses", "merge_k_lists", "swap_pairs", "reverse_k_group", "remove_duplicates",
            "remove_element", "find_index", "divide_integers", "substring_concat", "next_permutation",
            "longest_valid_parens", "search_rotated", "search_range", "search_insert", "valid_sudoku",
            "sudoku_solver", "count_and_say", "combination_sum", "combination_sum_ii", "first_missing_positive",
            "trapping_rain_water", "multiply_strings", "wildcard_matching", "jump_game_ii", "permutations",
            "permutations_ii", "rotate_image", "group_anagrams", "pow_x_n"]


@recipe("tomas-ruiz", "leetcode-solutions", "My solutions to coding interview problems, one file each",
        language="Python", topics=["leetcode", "algorithms", "interview-prep"], stars=57,
        about="Fifty small solution files committed one or two a day over several months: good for log -S, "
              "grep, shortlog, and finding when a solution changed.")
def _leetcode(b):
    b.commit("Initial commit", {"README.md": "# LeetCode solutions\n\n| # | Problem |\n|---|---|\n",
                                ".gitignore": GITIGNORES["python"]}, by="tomas-ruiz")
    for i, name in enumerate(PROBLEMS, 1):
        title = name.replace("_", " ").title()
        code = (f'"""{i}. {title}"""\n\n\nclass Solution:\n    def solve(self, nums, target=None):\n'
                f'        seen = {{}}\n        for i, x in enumerate(nums):\n            if target is not None and '
                f'target - x in seen:\n                return [seen[target - x], i]\n            seen[x] = i\n'
                f'        return []\n')
        if name == "two_sum":
            code = ('"""1. Two Sum: O(n^2) brute force."""\n\n\nclass Solution:\n    def twoSum(self, nums, target):\n'
                    '        for i in range(len(nums)):\n            for j in range(i + 1, len(nums)):\n'
                    '                if nums[i] + nums[j] == target:\n                    return [i, j]\n'
                    '        return []\n')
        b.commit(f"Solve {i}. {title}", {f"python/{i:04d}_{name}.py": code,
                                         "README.md": b.read("README.md") + f"| {i} | [{title}](python/{i:04d}_{name}"
                                                                            f".py) |\n"},
                 by="tomas-ruiz", hours=(8, 40))
        if i == 20:
            b.commit("Two Sum: use a hash map, O(n)", {"python/0001_two_sum.py": (
                '"""1. Two Sum: one pass with a hash map, O(n)."""\n\n\nclass Solution:\n    def twoSum(self, nums, '
                'target):\n        seen = {}\n        for i, x in enumerate(nums):\n            if target - x in '
                'seen:\n                return [seen[target - x], i]\n            seen[x] = i\n        return []\n')},
                by="tomas-ruiz")
    b.commit("Add study notes and diagrams", {"notes/patterns.md": md_doc(b.rng, "Patterns", ["two pointers",
                                                                                             "sliding window",
                                                                                             "backtracking"]),
                                              "notes/sliding_window.png": png(1_100_000)}, by="tomas-ruiz")


# ============================================================================ 15. open-learning/first-contributions

CONTRIB_NAMES = ["Aisha Bello", "Marco Rossi", "Hana Kim", "Lucas Silva", "Ivy Zhou", "Pedro Costa", "Sara Lind",
                 "Mehmet Arslan", "Nina Petrova", "Kenji Mori", "Fatima Zahra", "Oliver Smith", "Amara Obi",
                 "Leo Novak", "Mei Wong", "Jakub Kowalski", "Rosa Diaz", "Tariq Aziz", "Emma Dubois", "Ali Reza"]


@recipe("open-learning", "first-contributions", "Make your first open-source contribution: fork, branch, PR",
        language="Markdown", topics=["beginner-friendly", "first-timers", "open-source", "tutorial"], stars=4800,
        about="The classic first-PR exercise: fork the repo, clone your fork over SSH, add your name to "
              "Contributors.md on a branch, push and open a pull request. Twenty contributors' PRs are merged "
              "already; a few are still open.")
def _first_contrib(b):
    b.commit("Initial commit", {
        "README.md": readme("First Contributions", "Practise the whole contribution workflow on a repository "
                            "that welcomes it.", [
                                ("1. Fork", "Fork this repository to your account (`gh repo fork` works too)."),
                                ("2. Clone your fork", "```\ngit clone git@github.com:<you>/first-contributions.git"
                                                       "\n```"),
                                ("3. Branch", "```\ngit switch -c add-your-name\n```"),
                                ("4. Add your name", "Add a line `- Your Name` at the end of `Contributors.md`."),
                                ("5. Commit and push", "```\ngit commit -am \"Add <your name> to Contributors\"\n"
                                                       "git push -u origin add-your-name\n```"),
                                ("6. Open a pull request", "Open a PR against `open-learning/first-contributions`, "
                                                           "for example with `gh pr create`.")]),
        "Contributors.md": "# Contributors\n\n- Mira Chen\n",
        "assets/fork.png": png(520_000), "assets/clone.png": png(480_000), "assets/pr.png": png(610_000),
        "LICENSE": license_text("MIT", 2024, "open-learning")}, by="mira-chen")
    from .catalog import PEOPLE
    for i, name in enumerate(CONTRIB_NAMES, 1):
        login = name.split()[0].lower() + "-" + name.split()[1][0].lower()
        PEOPLE.setdefault(login, (name, f"{login}@users.noreply.github.com", "+0000"))
        branch = f"add-{name.split()[0].lower()}"
        b.branch(branch)
        b.edit("Contributors.md", lambda t, name=name: t + f"- {name}\n", f"Add {name} to Contributors",
               by=login, hours=(1, 20))
        b.switch("main")
        b.merge_pr(i + 1, branch, f"Add {name} to Contributors", by="mira-chen", author=login)
    for j, (name, login) in enumerate([("Yara Haddad", "yara-h"), ("Ola Nordmann", "ola-n")]):
        PEOPLE.setdefault(login, (name, f"{login}@users.noreply.github.com", "+0000"))
        branch = f"add-{name.split()[0].lower()}"
        b.branch(branch)
        b.edit("Contributors.md", lambda t, name=name: t + f"- {name}\n", f"Add {name} to Contributors", by=login)
        b.switch("main")
        b.open_pr(30 + j, branch, f"Add {name} to Contributors", login, head_owner=login)
        b.drop(branch)            # the branch lives in the contributor's fork, not here
