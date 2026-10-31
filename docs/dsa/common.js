// What every Part III tutorial shares: step text in the chosen language, the node declaration, rendering a tutorial's
// steps, the reference program (with Java's functions placed inside their classes), and the test programs.
//
// A tutorial file describes itself (options, complexity, diagram, function specs, structures, steps, tests) and
// DSACommon.make() turns that into what docs/dsa/page.js and scripts/dsa_selftest.mjs use.
(function (root) {
  "use strict";
  const DL = root.DSALang || (typeof require !== "undefined" ? require("./lang.js") : null);
  const { parseExpr, expr } = DL;

  const S = (say, ir, body) => (body ? { say, ir, body } : { say, ir });
  const fname = (n, lang) => (lang === "java" ? DL.camel(n) : n);
  const INT_T = { py: "int", cpp: "int", java: "int", rs: "i32" };
  const STR_T = { py: "str", cpp: "std::string", java: "String", rs: "String" };
  const VALUES = { int: [10, 20, 30, 40, 50, 60], generic: ["ant", "bee", "cat", "dog", "eel", "fox"] };
  const ABSENT = { int: 99, generic: "yak" };

  // ---------------------------------------------------------------- step text
  const NULLW = { py: "None", cpp: "nullptr", java: "null", rs: "ptr::null_mut()" };
  const TEXT_VARS = { L: "R:LinkedList", S: "R:Stack", Q: "R:Queue", H: "R:HashTable", A: "R:UnboundedArray", n: "N", p: "N", q: "N", e: "N", prev: "N", last: "N", d: "N",
    x: "T", k: "T", v: "V", i: "int", j: "int", t: "int", h: "int", m: "int", start: "N", end: "N" };
  const textCtx = (c) => ({ ...c, vars: { ...TEXT_VARS }, muts: new Set() });
  // {null}, {fn:name(args)}, {e:<expression>} and {s:<statement>} in step text become code in the chosen language
  function fmt(text, c) {
    if (!text) return "";
    return text.replace(/\{(null|fn:(\w+)(\([^)]*\))?|e:([^}]+)|s:([^}]+))\}/g, (m, a, fn, args, ex, st) => {
      if (a === "null") return "`" + NULLW[c.lang] + "`";
      if (fn) return "`" + fname(fn, c.lang) + (args || "") + "`";
      if (ex) return "`" + expr(parseExpr(ex), textCtx(c)) + "`";
      return "`" + DL.stmt(DL.parseStmt(st), textCtx(c)).code.join(" ") + "`";
    });
  }
  function fmtDeep(list, c) { for (const m of list) { m.say = fmt(m.say, c); fmtDeep(m.body, c); } }

  function contractNote(lang) {
    return {
      py: "Python runs `assert` unless started with `python -O`.",
      cpp: "C++ runs `assert` unless compiled with `-DNDEBUG`.",
      java: "Java skips `assert` unless started with `java -ea`; the checker turns assertions on.",
      rs: "`debug_assert!` runs in debug builds (what the checker uses) and disappears from release builds, like 15-122's contracts outside `-d` mode.",
    }[lang];
  }

  // ---------------------------------------------------------------- step 1 of every node-based structure
  // opts: { doubly, varName ("L"), rustWhy (why raw pointers) }
  function nodeSteps(c, opts) {
    const g = c.generic, L = c.lang, T = c.T, doubly = !!opts.doubly;
    if (L === "py") {
      const hint = g ? "Node[T] | None" : "Node | None";
      return [{ say: g ? "Declare the node class, generic over the element type: `class Node[T]`." : "Declare the node class `Node`.",
        code: [g ? "class Node[T]:" : "class Node:"], body: [
          { say: "Its constructor `__init__` takes the element.", code: [`def __init__(self, data: ${T}) -> None:`], body: [
            { say: "Store the element.", code: ["self.data = data"] },
            { say: "`next`: the following node, `None` (null) until something links it.", code: [`self.next: ${hint} = None`] },
            ...(doubly ? [{ say: "`prev`: the node before, also `None` for now.", code: [`self.prev: ${hint} = None`] }] : []),
          ] },
        ] }];
    }
    if (L === "cpp") {
      const P = g ? "Node<T>*" : "Node*";
      return [
        { say: "At the top of the file: `<iostream>` for printing, `<cassert>` for contracts.", code: ["#include <iostream>", "#include <cassert>"] },
        { say: g ? "Declare the node struct, as a template over the element type `T`." : "Declare the node struct `Node`.",
          code: g ? ["template <typename T>", "struct Node {"] : ["struct Node {"], close: ["};"], body: [
            { say: "A field for the element.", code: [`${T} data;`] },
            { say: "A pointer to the next node, null by default.", code: [`${P} next = nullptr;`] },
            ...(doubly ? [{ say: "A pointer to the previous node, null by default.", code: [`${P} prev = nullptr;`] }] : []),
            { say: "A constructor that stores the element; the pointers keep their null defaults.", code: [`Node(${T} data) : data(data) {}`] },
          ] },
      ];
    }
    if (L === "java") {
      const P = g ? "Node<T>" : "Node";
      return [{ say: g ? "Declare the class `Node<T>`, generic over the element type." : "Declare the class `Node`.",
        code: [g ? "class Node<T> {" : "class Node {"], close: ["}"], body: [
          { say: "A field for the element.", code: [`${T} data;`] },
          { say: "A reference to the next node. Java starts every reference field as `null`, so there's nothing else to write.", code: [`${P} next;`] },
          ...(doubly ? [{ say: "A reference to the previous node (also `null` at first).", code: [`${P} prev;`] }] : []),
          { say: "A constructor that stores the element.", code: [`Node(${T} data) {`], close: ["}"], body: [{ say: "", code: ["this.data = data;"] }] },
        ] }];
    }
    const N = g ? "Node<T>" : "Node";
    const init = doubly ? "Node { data, next: ptr::null_mut(), prev: ptr::null_mut() }" : "Node { data, next: ptr::null_mut() }";
    return [
      { say: `At the very top: allow capital variable names (the ${opts.what || "structure"} is called \`${opts.varName}\`), and import \`std::ptr\` for null pointers.`,
        code: ["#![allow(non_snake_case)]", "use std::ptr;"] },
      { say: g ? "Declare the struct `Node<T>`, generic over the element type." : "Declare the struct `Node`.",
        code: [`struct ${N} {`], close: ["}"], body: [
          { say: "A field for the element.", code: [`data: ${T},`] },
          { say: `A raw pointer to the next node (\`*mut\`). ${opts.rustWhy}`, code: [`next: *mut ${N},`] },
          ...(doubly ? [{ say: "A raw pointer to the previous node.", code: [`prev: *mut ${N},`] }] : []),
        ] },
      { say: "A constructor `Node::new` that puts the node on the heap and returns a raw pointer to it.",
        code: [g ? "impl<T> Node<T> {" : "impl Node {"], close: ["}"], body: [
          { say: "", code: [`fn new(data: ${T}) -> *mut ${N} {`], close: ["}"], body: [
            { say: "Box the node (a heap allocation) with null pointers, then turn the box into a raw pointer.", code: [`Box::into_raw(Box::new(${init}))`] }] },
        ] },
    ];
  }

  // ---------------------------------------------------------------- rendering
  // a tutorial's function specs and structures may depend on its options (a queue on a list, or on two stacks)
  const of = (x, o) => (typeof x === "function" ? x(o) : x);
  function context(def, o, overrides = {}) {
    const T = o.type === "int" ? INT_T[o.lang] : "T";
    return { lang: o.lang, generic: o.type === "generic", T, fns: of(def.FNS, o), structs: of(def.STRUCTS, o), vars: {}, muts: new Set(), contracts: contractsOn(o),
      ...(def.ctxExtra ? def.ctxExtra(o) : {}), ...overrides };
  }

  // rendered steps: [{id, title, goal, items: [display micro-steps], parts: [{code, cls|owner}], names}]
  function render(def, o, list = def.steps(o)) {
    const c = context(def, o);
    return list.map((st) => {
      const items = [];
      const parts = [];
      if (st.decl) {
        const d = DL.toDisplay(st.decl(c), c);
        fmtDeep(d, c);
        items.push(...d);
        d.forEach((m) => parts.push({ code: DL.emit([m]).join("\n"), cls: m.deferClose ? m.cls : null }));
      }
      for (const f of st.fns || []) {
        const r = DL.renderFunction(f, c);
        fmtDeep(r.display, c);
        items.push({ say: fmt(f.headerSay, c), group: true, fn: f.name, body: r.display, code: [], close: [] });
        parts.push({ code: r.code, owner: c.fns[f.name].owner });
      }
      return { id: st.id, title: st.title, goal: fmt(st.goal, c), items, parts, contractNote: st.contractNote ? contractNote(o.lang) : null,
        names: (st.fns || []).map((f) => fname(f.name, o.lang)) };
    });
  }

  // The reference solution for steps [0, upto): the user's file, as the tutorial builds it. In Java the functions are
  // static methods inside their structure's class, whose closing brace comes after them.
  function program(def, o, upto, list) {
    const parts = render(def, o, list).slice(0, upto).flatMap((s) => s.parts);
    if (o.lang !== "java") return parts.map((p) => p.code).join("\n\n") + "\n";
    const out = [];
    const classes = {};
    for (const p of parts) {
      if (p.cls) {
        const e = { head: p.code.slice(0, p.code.lastIndexOf("}")).replace(/\s+$/, ""), fns: [] };
        classes[p.cls] = e;
        out.push(e);
      } else if (p.owner && classes[p.owner]) classes[p.owner].fns.push(p.code);
      else out.push({ code: p.code });
    }
    return out.map((e) => (e.code !== undefined ? e.code
      : `${e.head}\n${e.fns.length ? "\n" + DL.indent(e.fns.join("\n\n"), 1) + "\n" : ""}}`)).join("\n\n") + "\n";
  }

  // the exact function headers the tutorial uses (Rust's generic bounds included), for the final problem statement
  function headers(def, o) {
    const c = context(def, o);
    const heads = {};
    (def.rawSteps || def.steps)(o).forEach((st) => (st.fns || []).forEach((f) => { heads[f.name] = DL.renderFunction(f, c).header; }));
    return (n) => heads[n].filter((l) => !l.startsWith("template")).join(" ").replace(/\s*[{:]$/, "");
  }

  // ---------------------------------------------------------------- tests
  // A group is a list of ops [kind, value]; handlers[kind](value, state, o) -> { ir: [statements], out: line(s) }.
  // Each group runs in its own block (a fresh structure); the out lines are what a correct solution prints.
  function compileGroups(groups, handlers) {
    const ir = [];
    const out = [];
    for (const g of groups) {
      const body = [];
      const state = {};
      for (const [kind, x] of g) {
        const h = handlers[kind];
        if (!h) throw new Error(`no test op ${kind}`);
        const r = h(x, state);
        body.push(...r.ir);
        if (r.out != null) out.push(...[].concat(r.out));
      }
      ir.push(S("", "block", body));
    }
    return { ir, expected: out.join("\n") + "\n" };
  }

  const lit = (x) => (typeof x === "number" ? String(x) : JSON.stringify(x));
  const say = (text) => S("", `print "${text}"`);
  // common op builders
  const ops = {
    // step 1's test: a fresh node
    node: (doubly) => (x) => ({
      ir: [S("", `let n = new(${lit(x)})`), say(`Node(${x}): data=`), S("", "print n.data"), say(", next is null: "), S("", "printb n.next == null"),
        ...(doubly ? [say(", prev is null: "), S("", "printb n.prev == null")] : []), S("", "println")],
      out: `Node(${x}): data=${x}, next is null: true${doubly ? ", prev is null: true" : ""}`,
    }),
    // print "label = <bool result>"
    bool: (o, call, label, value) => ({ ir: [say(`${label} = `), S("", `printb ${call}`), S("", "println")], out: `${label} = ${value}` }),
    // print "label = <value>"
    value: (o, call, label, value) => ({ ir: [say(`${label} = `), S("", `print ${call}`), S("", "println")], out: `${label} = ${value}` }),
  };

  function mulberry32(seed) {
    return () => {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // The test program for steps [0, upto) (or the final challenge): the pieces DSARun.assemble puts around the user's
  // code, and the expected output.
  function harness(def, o, upto, opts = {}) {
    const st = def.steps(o);
    const names = opts.final ? st[st.length - 1].tests : st[upto - 1].tests;
    const groups = withoutContractTests(def, o, def.testGroups(names, o, opts.final ? (opts.seed || 1) : 0));
    const { ir, expected } = compileGroups(groups, def.handlers(o));
    const refCtx = context(def, o, { ref: true, qual: true });
    const helpers = def.refFns(o, names).map((f) => DL.renderFunction(f, refCtx).code).join("\n\n");
    const extra = def.ctxExtra ? def.ctxExtra(o) : {};
    const mctx = context(def, o, { main: true, qual: true, T: extra.T || (o.type === "int" ? INT_T[o.lang] : STR_T[o.lang]) });
    return { helpers, main: DL.renderLines(ir, mctx).join("\n"), expected };
  }

  // ---------------------------------------------------------------- the Contracts option
  // Off (the default): no representation invariants or checks; just the structure and its operations. On: the
  // 15-122 way, with is_list(L) and friends, and checks at the start and end of every operation.
  const CONTRACTS = { id: "contracts", label: "Contracts", choices: [["off", "off"], ["on", "on"]], summary: { off: "no contracts", on: "contracts" },
    help: { off: "Write the structure and its operations, nothing else.",
      on: "Also write the contracts: the representation invariant (is_list, is_sorted, …) and checks at the start and end of every operation, as in CMU 15-122. The checker runs them." } };
  const contractsOn = (o) => o.contracts === "on";

  // Steps without contracts: contract-only functions (spec.contract) and every assert go; a step left with nothing to
  // write disappears, and its declarations (if any) move into the next step.
  function withoutContracts(def, o, list) {
    if (contractsOn(o)) return list;
    const fns = of(def.FNS, o);
    const strip = (steps) => steps.filter((st) => !(st.ir && /^assert\b/.test(st.ir))).map((st) => (st.body ? { ...st, body: strip(st.body) } : st));
    const out = [], gone = new Set();
    let carry = null;
    for (const st of list) {
      const kept = (st.fns || []).filter((f) => !fns[f.name].contract).map((f) => ({ ...f, steps: strip(f.steps) }));
      let decl = st.decl;
      if (carry) { const a = carry, b = decl; decl = b ? (c) => [...a(c), ...b(c)] : a; carry = null; }
      if ((st.fns || []).length && !kept.length) { if (decl) carry = decl; gone.add(st.id); continue; }
      out.push({ ...st, fns: kept, decl, contractNote: false });
    }
    return out.map((st) => ({ ...st, tests: st.tests.filter((n) => !gone.has(n)) }));
  }
  // ... and the tests that only exercise contracts
  function withoutContractTests(def, o, groups) {
    if (contractsOn(o)) return groups;
    const ops = new Set(def.CONTRACT_OPS || []);
    return groups.filter((g) => !g.some(([k]) => /break/.test(k))).map((g) => g.filter(([k]) => !ops.has(k)));
  }

  // a tutorial object for docs/dsa/page.js and the self-test
  function make(def) {
    const raw = def.steps;
    def.rawSteps = raw;   // (the final problem's signatures come from every function, contracts included)
    def.steps = (o) => withoutContracts(def, o, raw(o));
    const problem = def.problem;
    const t = {
      ...def,
      OPTIONS: [...def.OPTIONS, CONTRACTS],
      DEFAULTS: { ...def.DEFAULTS, contracts: "off" },
      // the boxes about contracts, and the contract functions and advice in the final problem, only with contracts on
      complexity: (o) => def.complexity(o).filter((c) => contractsOn(o) || !(c.id.startsWith("is_") || c.id === "contracts")),
      problem: (o) => {
        let text = problem(o);
        if (contractsOn(o)) return text;
        const fns = of(def.FNS, o);
        const names = Object.keys(fns).filter((n) => fns[n].contract).map((n) => fname(n, o.lang));
        text = text.split("\n").filter((l) => !(l.startsWith("* `") && names.some((n) => l.slice(0, l.indexOf("(")).endsWith(n)))).join("\n");
        return text.replace(/Check [^.]*\. /, "");
      },
      stepCount: (o) => def.steps(o).length,
      render: (o, list) => render(def, o, list),
      program: (o, upto, list) => program(def, o, upto ?? def.steps(o).length, list),
      harness: (o, upto, opts) => harness(def, o, upto, opts),
      context: (o, overrides) => context(def, o, overrides),
    };
    root.DSATutorials = root.DSATutorials || {};
    root.DSATutorials[def.id] = t;
    return t;
  }

  const api = { S, fname, fmt, fmtDeep, contractNote, contractsOn, nodeSteps, context, render, program, headers, compileGroups, ops, lit, say,
    mulberry32, harness, make, VALUES, ABSENT, INT_T, STR_T };
  root.DSACommon = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
