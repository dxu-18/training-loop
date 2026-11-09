"""Plausible file contents for the catalog's repositories: source files in a few languages, docs, data and the
usual project files (.gitignore, CI workflows, licenses). Everything is driven by a seeded random.Random, so the
same recipe always writes the same bytes."""

from __future__ import annotations

VERBS = ["load", "parse", "build", "render", "fetch", "merge", "split", "format", "validate", "normalize", "encode",
         "decode", "resolve", "compute", "update", "collect", "filter", "sort", "write", "read", "scan", "apply",
         "convert", "register", "lookup", "flush", "retry", "measure", "export", "import"]
NOUNS = ["config", "header", "token", "record", "payload", "session", "cache", "batch", "index", "chunk", "entry",
         "path", "schema", "report", "event", "metric", "query", "result", "frame", "buffer", "options", "user",
         "item", "row", "column", "request", "response", "node", "edge", "snapshot"]
ADJ = ["fast", "small", "safe", "simple", "lazy", "strict", "plain", "raw", "default", "local", "remote", "cached"]


def pick(rng, xs):
    return xs[rng.randrange(len(xs))]


def ident(rng, sep="_"):
    return f"{pick(rng, VERBS)}{sep}{pick(rng, NOUNS)}"


def camel(rng):
    return pick(rng, VERBS) + pick(rng, NOUNS).capitalize()


# ---------------------------------------------------------------------------- prose

OPENERS = ["This module", "The helper below", "This function", "The loader", "This step", "The parser"]
ACTIONS = ["keeps the {n} small", "returns a new {n} instead of changing the old one", "reads the {n} once and "
           "caches it", "checks the {n} before using it", "turns a {n} into plain text", "skips an empty {n}",
           "groups each {n} by its key", "falls back to the default {n}", "logs every {n} it rejects",
           "stops at the first bad {n}"]
WHY = ["so callers never see a half-written file", "which keeps the tests fast", "because the old code did this "
       "twice", "to match what users expect", "so the output is stable between runs", "which matters for big inputs",
       "to avoid a second network round trip", "and the result can be shared safely"]


def sentence(rng, topic=None):
    n = topic or pick(rng, NOUNS)
    return f"{pick(rng, OPENERS)} {pick(rng, ACTIONS).format(n=n)}, {pick(rng, WHY)}."


def paragraph(rng, topic=None, n=3):
    return " ".join(sentence(rng, topic) for _ in range(n))


def md_doc(rng, title, topics, sections=4):
    out = [f"# {title}", "", paragraph(rng, topics[0] if topics else None, 2), ""]
    for i in range(sections):
        t = topics[i % len(topics)] if topics else pick(rng, NOUNS)
        out += [f"## {t.capitalize()}", "", paragraph(rng, t, rng.randint(2, 4)), ""]
        if rng.random() < 0.5:
            out += ["```", f"{ident(rng)}({t})", "```", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------- code

def py_function(rng, name=None, doc=None):
    name = name or ident(rng)
    arg = pick(rng, NOUNS)
    other = pick(rng, [n for n in NOUNS if n != arg])
    body = pick(rng, [
        [f"if not {arg}:", f"    return None", f"return {{k: v for k, v in {arg}.items() if v is not None}}"],
        [f"out = []", f"for {other} in {arg}:", f"    if {other}:", f"        out.append({other})", "return out"],
        [f"{other} = str({arg}).strip()", f"return {other}.lower() if {other} else default"],
        [f"total = 0", f"for i, {other} in enumerate({arg}):", f"    total += len(str({other})) * (i + 1)",
         "return total"],
        [f"if isinstance({arg}, (list, tuple)):", f"    return [{name}(x) for x in {arg}]",
         f"return {arg}"],
        [f"seen = set()", f"for {other} in {arg}:", f"    if {other} in seen:", f"        raise ValueError("
         f"f\"duplicate {other}: {{{other}!r}}\")", f"    seen.add({other})", "return sorted(seen)"],
    ])
    need_default = any("default" in ln for ln in body)
    sig = f"def {name}({arg}{', default=None' if need_default else ''}):"
    doc = doc or sentence(rng, arg)
    return "\n".join([sig, f'    """{doc}"""'] + ["    " + ln for ln in body]) + "\n"


def py_module(rng, doc, n=4, names=(), imports=("os",), extra=""):
    parts = [f'"""{doc}"""', ""]
    parts += [f"import {m}" for m in imports]
    if imports:
        parts.append("")
    parts.append("")
    funcs = [py_function(rng, nm) for nm in names] + [py_function(rng) for _ in range(max(0, n - len(names)))]
    body = "\n\n".join(funcs)
    return "\n".join(parts) + body + ("\n\n" + extra if extra else "")


def py_test(rng, module, names):
    lines = [f"import unittest", "", f"import {module}", "", "", f"class Test{module.capitalize()}(unittest.TestCase):"]
    for nm in names:
        lines += [f"    def test_{nm}(self):", f"        self.assertTrue(callable({module}.{nm}))", ""]
    lines += ["", 'if __name__ == "__main__":', "    unittest.main()"]
    return "\n".join(lines) + "\n"


def js_module(rng, doc, n=4, esm=True):
    out = [f"// {doc}", ""]
    names = []
    for _ in range(n):
        nm = camel(rng)
        names.append(nm)
        arg = pick(rng, NOUNS)
        body = pick(rng, [
            [f"if (!{arg}) return null;", f"return {{ ...{arg}, updatedAt: Date.now() }};"],
            [f"return {arg}.filter(Boolean).map((x) => String(x).trim());"],
            [f"const out = new Map();", f"for (const x of {arg}) out.set(x.id, x);", "return out;"],
            [f"if (typeof {arg} !== 'string') throw new TypeError('{arg} must be a string');",
             f"return {arg}.toLowerCase();"],
        ])
        out += [f"{'export ' if esm else ''}function {nm}({arg}) {{"] + ["  " + b for b in body] + ["}", ""]
    if not esm:
        out.append("module.exports = { " + ", ".join(names) + " };")
    return "\n".join(out) + "\n"


def go_file(rng, pkg, doc, n=3):
    out = [f"// Package {pkg} {doc}", f"package {pkg}", "", 'import (', '\t"errors"', '\t"strings"', ")", ""]
    for _ in range(n):
        nm = camel(rng)
        nm = nm[0].upper() + nm[1:]
        arg = pick(rng, NOUNS)
        out += [f"// {nm} {pick(rng, ACTIONS).format(n=arg)}.", f"func {nm}({arg} string) (string, error) {{",
                f"\tif {arg} == \"\" {{", f"\t\treturn \"\", errors.New(\"empty {arg}\")", "\t}",
                f"\treturn strings.TrimSpace({arg}), nil", "}", ""]
    return "\n".join(out)


def rust_file(rng, doc, n=3):
    out = [f"//! {doc}", "", "use std::collections::HashMap;", ""]
    for _ in range(n):
        nm = ident(rng)
        arg = pick(rng, NOUNS)
        out += [f"/// {pick(rng, ACTIONS).format(n=arg).capitalize()}.", f"pub fn {nm}({arg}: &[String]) -> "
                f"HashMap<String, usize> {{", "    let mut counts = HashMap::new();", f"    for item in {arg} {{",
                "        *counts.entry(item.clone()).or_insert(0) += 1;", "    }", "    counts", "}", ""]
    return "\n".join(out)


def c_file(rng, doc, n=3, header=None):
    out = [f"/* {doc} */", "#include <stdio.h>", "#include <string.h>"]
    if header:
        out.append(f'#include "{header}"')
    out.append("")
    for _ in range(n):
        nm = ident(rng)
        arg = pick(rng, NOUNS)
        out += [f"/* {pick(rng, ACTIONS).format(n=arg)} */", f"int {nm}(const char *{arg}, size_t len)", "{",
                "\tsize_t i;", "\tint n = 0;", "", f"\tif (!{arg})", "\t\treturn -1;", "\tfor (i = 0; i < len; i++)",
                f"\t\tif ({arg}[i] == '\\n')", "\t\t\tn++;", "\treturn n;", "}", ""]
    return "\n".join(out)


def csv_text(rng, header, row, n):
    return ",".join(header) + "\n" + "".join(",".join(str(x) for x in row(rng, i)) + "\n" for i in range(n))


# ---------------------------------------------------------------------------- project files

def license_text(kind, year, holder):
    url = {"MIT": "https://opensource.org/license/mit", "Apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
           "BSD-3-Clause": "https://opensource.org/license/bsd-3-clause", "GPL-2.0": "https://www.gnu.org/licenses/"
           "old-licenses/gpl-2.0.html"}.get(kind, "")
    return (f"{kind} License\n\nCopyright (c) {year} {holder}\n\nThis project is released under the {kind} license.\n"
            f"The full license text is at {url}\n")


GITIGNORES = {
    "python": "__pycache__/\n*.py[cod]\n.venv/\nvenv/\n.env\n*.egg-info/\ndist/\nbuild/\n.pytest_cache/\n.ipynb_checkpoints/\n",
    "node": "node_modules/\ndist/\n.env\n.env.local\nnpm-debug.log*\ncoverage/\n.DS_Store\n",
    "go": "/bin/\n*.exe\n*.test\n*.out\nvendor/\n",
    "rust": "/target/\nCargo.lock.bak\n**/*.rs.bk\n",
    "c": "*.o\n*.a\n*.so\n/build/\n/jq-mini\ncore\n",
    "latex": "*.aux\n*.log\n*.out\n*.toc\n*.bbl\n*.blg\n*.synctex.gz\n*.fdb_latexmk\n*.fls\nbuild/\n",
    "ml": "__pycache__/\n.venv/\n.env\ndata/raw/\noutputs/\nwandb/\ncheckpoints/\n*.ckpt\n.ipynb_checkpoints/\n",
    "terraform": ".terraform/\n*.tfstate\n*.tfstate.*\n.terraform.lock.hcl\ncrash.log\n*.tfvars\n",
    "web": ".DS_Store\n_site/\n.sass-cache/\nnode_modules/\n",
}


def ci_workflow(lang, name="CI"):
    steps = {
        "python": ["      - uses: actions/setup-python@v5", "        with:", "          python-version: \"3.12\"",
                   "      - run: pip install -r requirements.txt", "      - run: python -m pytest -q"],
        "node": ["      - uses: actions/setup-node@v4", "        with:", "          node-version: 20",
                 "      - run: npm ci", "      - run: npm test"],
        "go": ["      - uses: actions/setup-go@v5", "        with:", "          go-version: \"1.22\"",
               "      - run: go test ./..."],
        "rust": ["      - run: cargo build --verbose", "      - run: cargo test --verbose"],
        "c": ["      - run: make", "      - run: make test"],
    }[lang]
    return "\n".join([f"name: {name}", "", "on:", "  push:", "    branches: [main]", "  pull_request:", "",
                      "jobs:", "  test:", "    runs-on: ubuntu-latest", "    steps:", "      - uses: actions/checkout@v4"]
                     + steps) + "\n"


def png(size):
    from .catalog import Lazy
    return Lazy(size, b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")


def jpg(size):
    from .catalog import Lazy
    return Lazy(size, b"\xff\xd8\xff\xe0\x00\x10JFIF\x00")


def pdf(size):
    from .catalog import Lazy
    return Lazy(size, b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n")


def binary(size, head=b""):
    from .catalog import Lazy
    return Lazy(size, head)
