// Running tutorial code. Python runs in the page's Pyodide worker (generator/api.py: dsa_run); C++, Java and Rust are
// compiled and run on Compiler Explorer (godbolt.org), whose API accepts requests from any site.
// assemble() puts the checker's tests around the user's file; compare() finds the first line that differs.
(function (root) {
  "use strict";
  const DL = root.DSALang || (typeof require !== "undefined" ? require("./lang.js") : null);

  const CE_URL = "https://godbolt.org/api/compiler";
  const COMPILERS = {
    cpp: { id: "g152", args: "-std=c++20 -O1", name: "GCC 15.2" },
    java: { id: "java2500", args: "", name: "JDK 25" },
    rs: { id: "r1940", args: "--edition 2021", name: "rustc 1.94" },
  };

  const hasMain = {
    py: () => true,
    cpp: (s) => /\bint\s+main\s*\(/.test(s),
    java: (s) => /\bstatic\s+void\s+main\s*\(/.test(s),
    rs: (s) => /\bfn\s+main\s*\(/.test(s),
  };

  // The user's file plus the checker: the user's own main (if any) is renamed so the checker's runs instead.
  // tests = { helpers, main } from a tutorial's harness(); null = run the user's file by itself.
  function assemble(lang, user, tests) {
    if (lang === "py") {
      // printb's helper: a function that falls off its end returns None, which must not pass for false
      const bool = 'def tl_bool(v):\n    return "true" if v is True else "false" if v is False else repr(v)\n';
      return { user, tests: tests ? `${bool}\n${tests.helpers}\n\n${tests.main}\n` : "" };
    }
    if (!tests) return { source: lang === "java" ? user.replace(/^(\s*)public\s+((?:final\s+|abstract\s+)?class\b)/gm, "$1$2") : user };
    // Nothing goes before the user's code, so line numbers in errors match the editor.
    if (lang === "cpp") {
      const u = user.replace(/\bint(\s+)main\s*\(/g, "int$1tl_user_main(");
      return { source: `${u}\n\n// ---- the checker's tests\n#include <iostream>\n#include <string>\n#include <cassert>\n#include <vector>\n#include <utility>\n${tests.helpers}\n\n` +
        `int main() {\n    std::cout << std::unitbuf;\n${DL.indent(tests.main, 1)}\n    return 0;\n}\n` };
    }
    if (lang === "java") {
      const u = user.replace(/^(\s*)public\s+((?:final\s+|abstract\s+)?class\b)/gm, "$1$2")
        .replace(/\bpublic\s+static\s+void\s+main\s*\(/g, "static void tlUserMain(").replace(/\bstatic\s+void\s+main\s*\(/g, "static void tlUserMain(");
      // assertions on before any of the user's classes is loaded (they load on first use)
      return { source: `${u}\n\n// ---- the checker's tests\nclass TLTests {\n${DL.indent(tests.helpers, 1)}\n\n` +
        `    public static void main(String[] args) {\n        ClassLoader.getSystemClassLoader().setDefaultAssertionStatus(true);\n` +
        `${DL.indent(tests.main, 2)}\n    }\n}\n` };
    }
    // Rust (warnings, e.g. about the capital L, don't stop the build and aren't shown)
    const u = user.replace(/\bfn(\s+)main\s*\(/g, "fn$1tl_user_main(");
    return { source: `${u}\n\n// ---- the checker's tests\n${tests.helpers}\n\nfn main() {\n    unsafe {\n${DL.indent(tests.main, 2)}\n    }\n}\n` };
  }

  // -> { compiled, stdout, stderr, compileOutput, exit, timedOut }
  async function runCE(lang, source, fetchImpl) {
    const c = COMPILERS[lang];
    const res = await (fetchImpl || fetch)(`${CE_URL}/${c.id}/compile`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({
        source, allowStoreCodeDebug: false,
        options: { userArguments: c.args, executeParameters: { args: [], stdin: "" },
          compilerOptions: { executorRequest: true, skipAsm: true }, filters: { execute: true }, tools: [], libraries: [] },
      }),
    });
    if (!res.ok) throw new Error(`Compiler Explorer answered ${res.status} ${res.statusText}`.trim());
    const d = await res.json();
    const text = (a) => (a || []).map((x) => x.text).join("\n");
    const build = d.buildResult || {};
    if (!d.didExecute) {
      return { compiled: false, compileOutput: text(build.stderr) || text(build.stdout) || text(d.stderr) || "The build failed.", stdout: "", stderr: "" };
    }
    return { compiled: true, stdout: text(d.stdout), stderr: text(d.stderr), exit: d.code, timedOut: !!d.timedOut };
  }

  // Compare the program's output with the expected lines. -> { ok, passed: [lines], expected, got, line }
  function compare(expected, actual) {
    const E = expected.replace(/\n+$/, "").split("\n").map((l) => l.replace(/\s+$/, ""));
    const A = (actual || "").replace(/\s+$/, "").split("\n").map((l) => l.replace(/\s+$/, ""));
    if (A.length === 1 && A[0] === "") A.length = 0;
    for (let i = 0; i < E.length; i++) {
      if (i >= A.length) return { ok: false, passed: E.slice(0, i), expected: E[i], got: null, line: i };
      if (A[i] !== E[i]) return { ok: false, passed: E.slice(0, i), expected: E[i], got: A[i], line: i };
    }
    if (A.length > E.length) return { ok: false, passed: E, expected: null, got: A[E.length], line: E.length };
    return { ok: true, passed: E };
  }

  const api = { COMPILERS, assemble, runCE, compare, hasMain };
  root.DSARun = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
