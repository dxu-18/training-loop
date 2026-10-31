// Data Structures & Algorithms tutorials: a tiny statement language rendered to Python, C++, Java and Rust.
//
// A tutorial writes each function once, as micro-steps whose code is one statement of this language
// ("n.next = L.head", "if L.tail == null", "let x = take n"). Rendering one source gives the code every
// micro-step shows, the reference solution the self-test compiles (scripts/dsa_selftest.mjs), and the test
// programs the Check button runs. Types are inferred: N = node pointer, T = element, int, bool, and the tutorial's
// structures (ctx.structs: {Stack: {fields}}): R:Stack = a reference parameter, V:Stack = a value (a local variable in
// the tests' main, or a field such as a queue's two stacks).
(function (root) {
  "use strict";

  const LANGS = {
    py: { id: "py", name: "Python", mode: "python", cLike: false },
    cpp: { id: "cpp", name: "C++", mode: "text/x-c++src", cLike: true },
    java: { id: "java", name: "Java", mode: "text/x-java", cLike: true },
    rs: { id: "rs", name: "Rust", mode: "rust", cLike: true },
  };
  const camel = (s) => s.replace(/_([a-z])/g, (_, c) => c.toUpperCase());

  // ---------------------------------------------------------------- parsing
  const TOKEN = /\s*(\d+|"(?:[^"\\]|\\.)*"|[A-Za-z_]\w*|==|!=|<=|>=|[-+<>().,])/y;
  function tokenize(src) {
    const out = [];
    TOKEN.lastIndex = 0;
    while (TOKEN.lastIndex < src.length) {
      const at = TOKEN.lastIndex;
      const m = TOKEN.exec(src);
      if (!m) {
        if (/^\s*$/.test(src.slice(at))) break;
        throw new Error(`IR: can't read ${JSON.stringify(src.slice(at))} in ${JSON.stringify(src)}`);
      }
      out.push(m[1]);
    }
    return out;
  }

  function parseExpr(src) {
    const t = tokenize(src);
    let i = 0;
    const peek = () => t[i];
    const next = () => t[i++];
    const expect = (s) => { if (t[i] !== s) throw new Error(`IR: expected ${s} in ${JSON.stringify(src)}`); i++; };
    const or = () => { let a = and(); while (peek() === "or") { next(); a = { k: "bin", op: "or", a, b: and() }; } return a; };
    const and = () => { let a = not(); while (peek() === "and") { next(); a = { k: "bin", op: "and", a, b: not() }; } return a; };
    const not = () => { if (peek() === "not") { next(); return { k: "not", a: not() }; } return cmp(); };
    const cmp = () => {
      const a = add();
      if (["==", "!=", "<", ">", "<=", ">="].includes(peek())) { const op = next(); return { k: "bin", op, a, b: add() }; }
      return a;
    };
    const add = () => {
      let a = post();
      while (peek() === "+" || peek() === "-") { const op = next(); a = { k: "bin", op, a, b: post() }; }
      return a;
    };
    const post = () => { let a = prim(); while (peek() === ".") { next(); a = { k: "field", obj: a, name: next() }; } return a; };
    const prim = () => {
      const s = next();
      if (s === undefined) throw new Error(`IR: unexpected end of ${JSON.stringify(src)}`);
      if (s === "(") { const e = or(); expect(")"); return e; }
      if (/^\d/.test(s)) return { k: "num", v: Number(s) };
      if (s[0] === '"') return { k: "str", v: JSON.parse(s) };
      if (s === "null") return { k: "null" };
      if (s === "true" || s === "false") return { k: "bool", v: s === "true" };
      if (s === "default") return { k: "default" };
      if (peek() === "(") {
        next();
        const args = [];
        if (peek() !== ")") { args.push(or()); while (peek() === ",") { next(); args.push(or()); } }
        expect(")");
        return s === "new" ? { k: "new", arg: args[0] } : { k: "call", fn: s, args };
      }
      return { k: "id", name: s };
    };
    const e = or();
    if (i !== t.length) throw new Error(`IR: trailing ${t.slice(i).join(" ")} in ${JSON.stringify(src)}`);
    return e;
  }

  function parseStmt(line) {
    let m;
    if ((m = line.match(/^let (\w+) = take (.+)$/))) return { k: "take", name: m[1], e: parseExpr(m[2]) };
    if ((m = line.match(/^let (\w+) = (.+)$/))) return { k: "let", name: m[1], e: parseExpr(m[2]) };
    if ((m = line.match(/^(if|elif|while|repeat) (.+)$/))) return { k: m[1], e: parseExpr(m[2]) };
    if (line === "else" || line === "block") return { k: line };
    if ((m = line.match(/^return(?: (.+))?$/))) return { k: "return", e: m[1] ? parseExpr(m[1]) : null };
    if ((m = line.match(/^(assert|free) (.+)$/))) return { k: m[1], e: parseExpr(m[2]) };
    if ((m = line.match(/^(print|println|printb)(?: (.+))?$/))) return { k: m[1], e: m[2] ? parseExpr(m[2]) : null };
    if ((m = line.match(/^make (\w+) (\w+)$/))) return { k: "make", struct: m[1], name: m[2] };
    if ((m = line.match(/^(.+?) (\+=|-=|=) (.+)$/))) return { k: "assign", lhs: parseExpr(m[1]), op: m[2], e: parseExpr(m[3]) };
    return { k: "expr", e: parseExpr(line) };
  }

  // ---------------------------------------------------------------- types
  const NODE_FIELDS = { data: "T", next: "N", prev: "N" };
  const isStruct = (t) => typeof t === "string" && (t.startsWith("R:") || t.startsWith("V:"));

  function typeOf(e, ctx) {
    switch (e.k) {
      case "num": return "int";
      case "str": return "T";
      case "null": return "null";
      case "bool": return "bool";
      case "default": return "T";
      case "new": return "N";
      case "id": {
        const t = ctx.vars[e.name];
        if (!t) throw new Error(`IR: unknown variable ${e.name}`);
        return t;
      }
      case "field": {
        const ot = typeOf(e.obj, ctx);
        const t = ot === "N" ? NODE_FIELDS[e.name] : isStruct(ot) ? ((ctx.structs[ot.slice(2)] || {}).fields || {})[e.name] : null;
        if (!t) throw new Error(`IR: no field ${e.name} on ${ot}`);
        return t;
      }
      case "call": { const r = spec(e.fn, ctx).ret; return r === "Tpeek" ? "T" : r; }
      case "bin": return e.op === "+" || e.op === "-" ? "int" : "bool";
      case "not": return "bool";
    }
    throw new Error(`IR: bad expression ${e.k}`);
  }

  function spec(name, ctx) {
    const s = ctx.fns[name.replace(/^ref_/, "")];
    if (!s) throw new Error(`IR: unknown function ${name}`);
    return s;
  }

  // ---------------------------------------------------------------- expressions
  const PREC = { or: 1, and: 2, "==": 4, "!=": 4, "<": 4, ">": 4, "<=": 4, ">=": 4, "+": 5, "-": 5 };

  function fnName(name, ctx) {
    const isRef = name.startsWith("ref_") || (ctx.ref && !spec(name, ctx).harness);
    const base = name.replace(/^ref_/, "");
    const full = (isRef ? "ref_" : "") + base;
    return ctx.lang === "java" ? camel(full) : full;
  }

  function nullLit(ctx) {
    return { py: "None", cpp: "nullptr", java: "null", rs: ctx.qual ? "std::ptr::null_mut()" : "ptr::null_mut()" }[ctx.lang];
  }

  function dataLit(v, ctx) {
    if (typeof v === "number") return String(v);
    const s = JSON.stringify(v);
    return { py: s, cpp: `std::string(${s})`, java: `new String(${s})`, rs: `String::from(${s})` }[ctx.lang];
  }

  function defaultLit(ctx) {
    if (!ctx.generic) return "0";
    return { py: "None", cpp: `${ctx.T}{}`, java: "null", rs: `${ctx.T}::default()` }[ctx.lang];
  }

  function newNode(arg, ctx) {
    const a = expr(arg, ctx);
    const g = ctx.generic;
    return { py: `Node(${a})`, cpp: g ? `new Node<${ctx.T}>(${a})` : `new Node(${a})`,
      java: g ? `new Node<>(${a})` : `new Node(${a})`, rs: `Node::new(${a})` }[ctx.lang];
  }

  function expr(e, ctx, need = 0) {
    const L = ctx.lang;
    const wrap = (s, p) => (p < need ? `(${s})` : s);
    switch (e.k) {
      case "num": return String(e.v);
      case "str": return dataLit(e.v, ctx);
      case "null": return nullLit(ctx);
      case "bool": return L === "py" ? (e.v ? "True" : "False") : String(e.v);
      case "default": return defaultLit(ctx);
      case "new": return newNode(e.arg, ctx);
      case "id": return e.name;
      case "field": {
        const ot = typeOf(e.obj, ctx);
        const o = expr(e.obj, ctx, 9);
        if (ot === "N" && L === "cpp") return `${o}->${e.name}`;
        if (ot === "N" && L === "rs") return `(*${o}).${e.name}`;
        return `${o}.${e.name}`;
      }
      case "call": {
        const s = spec(e.fn, ctx);
        const args = e.args.map((a, i) => arg(a, s.params[i][1], ctx));
        // Java: a static method of another class is called through it (LinkedList.insertFront(L, x))
        const qual = L === "java" && s.owner && s.owner !== ctx.owner && !e.fn.startsWith("ref_") && !ctx.ref && !s.harness ? `${s.owner}.` : "";
        // C++: a template can't deduce T from nullptr, so name it
        const targs = L === "cpp" && ctx.generic && e.args.some((a) => a.k === "null") ? `<${ctx.T}>` : "";
        return `${qual}${fnName(e.fn, ctx)}${targs}(${args.join(", ")})`;
      }
      case "not": {
        const INV = { "==": "!=", "!=": "==", "<": ">=", ">": "<=", "<=": ">", ">=": "<" };
        if (e.a.k === "bin" && INV[e.a.op]) return expr({ ...e.a, op: INV[e.a.op] }, ctx, need);   // not (a == b) -> a != b
        const inner = expr(e.a, ctx, L === "py" ? 3 : 9);
        return wrap(L === "py" ? `not ${inner}` : `!${inner}`, 3);
      }
      case "bin": {
        if (e.op === "and" || e.op === "or") {
          const p = PREC[e.op];
          const side = (x) => expr(x, ctx, x.k === "bin" && (x.op === "and" || x.op === "or") && x.op !== e.op ? 9 : p);
          const op = L === "py" ? e.op : e.op === "and" ? "&&" : "||";
          return wrap(`${side(e.a)} ${op} ${side(e.b)}`, p);
        }
        if (e.op === "==" || e.op === "!=") return wrap(compare(e, ctx), 4);
        const p = PREC[e.op];
        return wrap(`${expr(e.a, ctx, p)} ${e.op} ${expr(e.b, ctx, p + 1)}`, p);
      }
    }
    throw new Error(`IR: bad expression ${e.k}`);
  }

  function compare(e, ctx) {
    const L = ctx.lang, eq = e.op === "==";
    const ta = typeOf(e.a, ctx), tb = typeOf(e.b, ctx);
    if (ta === "null" || tb === "null") {
      const x = expr(ta === "null" ? e.b : e.a, ctx, 9);
      if (L === "py") return `${x} ${eq ? "is" : "is not"} None`;
      if (L === "rs") return `${eq ? "" : "!"}${x}.is_null()`;
      return `${x} ${e.op} ${nullLit(ctx)}`;
    }
    const a = expr(e.a, ctx, 5), b = expr(e.b, ctx, 5);
    if (ta === "N" && L === "py") return `${a} ${eq ? "is" : "is not"} ${b}`;
    if (ta === "T" && L === "java" && ctx.generic) return `${eq ? "" : "!"}${expr(e.a, ctx, 9)}.equals(${b})`;
    return `${a} ${e.op} ${b}`;
  }

  // a call argument: Rust borrows a structure held by value (a local, or a field like Q.inbox)
  function arg(a, kind, ctx) {
    const s = expr(a, ctx);
    if (ctx.lang === "rs" && String(typeOf(a, ctx)).startsWith("V:")) return kind.startsWith("mut:") ? `&mut ${s}` : `&${s}`;
    return s;
  }

  // ---------------------------------------------------------------- type names
  function typeName(t, ctx) {
    const g = ctx.generic ? `<${ctx.T}>` : "";
    if (t === "N") return ctx.lang === "cpp" ? `Node${g}*` : `Node${g}`;
    if (t === "T") return ctx.T;
    if (t === "int") return "int";
    if (t === "bool") return ctx.lang === "java" ? "boolean" : "bool";
    throw new Error(`IR: can't declare a ${t}`);
  }

  // ---------------------------------------------------------------- statements
  // stmt -> { code: [lines], close: [lines] | null (a block), note } ; lines are unindented
  function stmt(s, ctx) {
    const L = ctx.lang;
    const semi = L === "py" ? "" : ";";
    const cond = (e) => (L === "py" || L === "rs" ? expr(e, ctx) : `(${expr(e, ctx)})`);
    const open = (head) => (L === "py" ? `${head}:` : `${head} {`);
    const end = LANGS[L].cLike ? ["}"] : [];   // a block's closing line (an if's merges into a following else)
    switch (s.k) {
      case "let": {
        const t = typeOf(s.e, ctx);
        const val = expr(s.e, ctx);
        ctx.vars[s.name] = t === "null" ? "N" : t;
        const vt = ctx.vars[s.name];
        if (L === "py") return { code: [`${s.name} = ${val}`] };
        if (L === "rs") return { code: [`let ${ctx.muts.has(s.name) ? "mut " : ""}${s.name} = ${val};`] };
        return { code: [`${typeName(vt, ctx)} ${s.name} = ${val};`] };
      }
      case "take": {
        const n = expr(s.e, ctx, 9);
        ctx.vars[s.name] = "T";
        const field = expr({ k: "field", obj: s.e, name: "data" }, ctx);
        if (L === "py") return { code: [`${s.name} = ${field}`] };
        if (L === "rs") return { code: [`let ${s.name} = Box::from_raw(${n}).data;`] };
        if (L === "java") return { code: [`${ctx.T} ${s.name} = ${field};`] };
        return { code: [`${ctx.T} ${s.name} = ${field};`, `delete ${n};`] };
      }
      case "assign": {
        const lhs = expr(s.lhs, ctx);
        return { code: [`${lhs} ${s.op} ${expr(s.e, ctx)}${semi}`] };
      }
      case "if": return { code: [open(`if ${cond(s.e)}`)], close: end };
      case "elif": return { code: [L === "py" ? `elif ${expr(s.e, ctx)}:` : `} else if ${cond(s.e)} {`], close: end };
      case "else": return { code: [L === "py" ? "else:" : "} else {"], close: end };
      case "while": return { code: [open(`while ${cond(s.e)}`)], close: end };
      case "repeat": {
        const n = expr(s.e, ctx);
        const head = { py: `for _ in range(${n}):`, cpp: `for (int k = 0; k < ${n}; k++) {`,
          java: `for (int k = 0; k < ${n}; k++) {`, rs: `for _ in 0..${n} {` }[L];
        return { code: [head], close: end };
      }
      case "block": return { code: L === "py" ? [] : ["{"], close: end };
      case "return": {
        if (!s.e) return { code: [`return${semi}`] };
        // Rust's generic peek returns a reference to the element (it can't move it out of the node)
        const borrow = L === "rs" && ctx.generic && ctx.ret === "Tpeek" && s.e.k === "field" ? "&" : "";
        return { code: [`return ${borrow}${expr(s.e, ctx)}${semi}`] };
      }
      case "assert": {
        if (ctx.ref) return { code: [] };
        const c = expr(s.e, ctx);
        return { code: [{ py: `assert ${c}`, cpp: `assert(${c});`, java: `assert ${c};`, rs: `debug_assert!(${c});` }[L]] };
      }
      case "free": {
        const n = expr(s.e, ctx, 9);
        if (L === "cpp") return { code: [`delete ${n};`] };
        if (L === "rs") return { code: [`drop(Box::from_raw(${n}));`] };
        return { code: [], note: `Nothing to write in ${LANGS[L].name}: once nothing points to \`${n}\`, the garbage collector frees it.` };
      }
      case "print": case "println": case "printb": return { code: [printStmt(s, ctx)] };
      case "make": {
        const N = s.struct;
        ctx.vars[s.name] = `V:${N}`;
        const g = ctx.generic ? `<${ctx.T}>` : "";
        return { code: [{ py: `${s.name} = ${N}()`, cpp: `${N}${g} ${s.name};`,
          java: `${N}${g} ${s.name} = new ${N}${ctx.generic ? "<>" : ""}();`,
          rs: `let mut ${s.name}${ctx.generic ? `: ${N}${g}` : ""} = ${N}::new();` }[L]] };
      }
      case "expr": return { code: [`${expr(s.e, ctx)}${semi}`] };
    }
    throw new Error(`IR: bad statement ${s.k}`);
  }

  function printStmt(s, ctx) {
    const L = ctx.lang;
    const lit = s.e && s.e.k === "str";
    const text = lit ? JSON.stringify(s.e.v) : null;
    const rsText = lit ? text.replace(/[{}]/g, (c) => c + c) : null;
    if (s.k === "printb") {
      const c = expr(s.e, ctx);
      return { py: `print(tl_bool(${c}), end="")`, cpp: `std::cout << (${c} ? "true" : "false");`,
        java: `System.out.print(${c});`, rs: `print!("{}", ${c});` }[L];
    }
    const nl = s.k === "println";
    if (!s.e) return { py: "print()", cpp: 'std::cout << "\\n";', java: "System.out.println();", rs: "println!();" }[L];
    if (lit) {
      return { py: nl ? `print(${text})` : `print(${text}, end="")`, cpp: nl ? `std::cout << ${text.slice(0, -1)}\\n";` : `std::cout << ${text};`,
        java: `System.out.${nl ? "println" : "print"}(${text});`, rs: `${nl ? "println" : "print"}!(${rsText});` }[L];
    }
    const v = expr(s.e, ctx);
    return { py: nl ? `print(${v})` : `print(${v}, end="")`, cpp: nl ? `std::cout << ${v} << "\\n";` : `std::cout << ${v};`,
      java: `System.out.${nl ? "println" : "print"}(${v});`, rs: `${nl ? "println" : "print"}!("{}", ${v});` }[L];
  }

  // ---------------------------------------------------------------- micro-steps
  // A micro-step is { say, ir | code, body?, close?, note? }. `ir` is one statement; `code` is raw lines for one
  // language ({py: [...], cpp: [...], ...}). toDisplay turns a list of them into rendered micro-steps:
  //   { say, code: [lines], body: [rendered...], close: [lines], note }
  function toDisplay(steps, ctx) {
    const out = steps.filter((st) => !st.only || st.only.includes(ctx.lang)).map((st) => {
      let r;
      if (st.ir !== undefined) {
        r = stmt(parseStmt(st.ir), ctx);
      } else {
        const raw = typeof st.code === "function" ? st.code(ctx) : st.code;
        const c = raw && (Array.isArray(raw) ? raw : raw[ctx.lang]);
        const cl = st.close && (Array.isArray(st.close) ? st.close : st.close[ctx.lang]);
        r = { code: c || [], close: st.body ? (cl || []) : cl || null };
      }
      const body = st.body ? toDisplay(st.body, ctx) : [];
      const say = typeof st.say === "function" ? st.say(ctx) : st.say || "";
      return { say, code: r.code, body, close: r.close || [], note: r.note || st.note || null, isElse: st.ir && /^(else|elif)\b/.test(st.ir),
        deferClose: !!st.deferClose, cls: st.cls || null };
    });
    // C-like: an if's "}" merges into the following "} else {"
    for (let i = 0; i + 1 < out.length; i++) if (out[i + 1].isElse && LANGS[ctx.lang].cLike) out[i].close = [];
    return out;
  }

  function emit(display, depth = 0, lines = []) {
    const pad = (d) => "    ".repeat(d);
    for (const m of display) {
      m.code.forEach((l) => lines.push(l ? pad(depth) + l : l));
      emit(m.body, m.code.length ? depth + 1 : depth, lines);   // (Python's `block` has no line of its own)
      m.close.forEach((l) => lines.push(pad(depth) + l));
    }
    return lines;
  }

  // names assigned to after their declaration (Rust needs `let mut` for them)
  function reassigned(steps, acc = new Set()) {
    for (const st of steps) {
      if (st.ir) {
        const m = st.ir.match(/^(\w+) (=|\+=|-=) /);
        if (m) acc.add(m[1]);
      }
      if (st.body) reassigned(st.body, acc);
    }
    return acc;
  }

  // what a Rust generic function needs from T, read from its code
  function rustBounds(steps, ctx) {
    const need = new Set();
    const walkE = (e, c) => {
      if (!e || typeof e !== "object") return;
      if (e.k === "default") need.add("Default");
      if (e.k === "bin" && (e.op === "==" || e.op === "!=") && typeOf(e.a, c) === "T") need.add("PartialEq");
      if (e.k === "call") { const s = spec(e.fn, c); (s.bounds || []).forEach((b) => need.add(b)); }
      for (const v of Object.values(e)) if (v && typeof v === "object") walkE(v, c);
    };
    const c = { ...ctx, vars: { ...ctx.vars } };
    const walk = (list) => {
      for (const st of list) {
        if (st.ir !== undefined && (!st.only || st.only.includes("rs"))) {
          const s = parseStmt(st.ir);
          walkE(s.e, c); walkE(s.lhs, c);
          if ((s.k === "print" || s.k === "println") && s.e && s.e.k !== "str" && typeOf(s.e, c) === "T") need.add("std::fmt::Display");
          if (s.k === "let") c.vars[s.name] = typeOf(s.e, c) === "null" ? "N" : typeOf(s.e, c);
          if (s.k === "take") c.vars[s.name] = "T";
        }
        if (st.body) walk(st.body);
      }
    };
    walk(steps);
    return ["PartialEq", "std::fmt::Display", "Default"].filter((b) => need.has(b));
  }

  // does a function follow a node pointer (Rust then needs an unsafe block)?
  function usesPointers(steps, ctx0) {
    const c = { ...ctx0, vars: { ...ctx0.vars } };
    let found = false;
    const walkE = (e) => {
      if (!e || typeof e !== "object" || found) return;
      if (e.k === "field" && typeOf(e.obj, c) === "N") found = true;
      for (const v of Object.values(e)) if (v && typeof v === "object") walkE(v);
    };
    const walk = (list) => {
      for (const st of list) {
        if (found) return;
        if (st.ir !== undefined && (!st.only || st.only.includes("rs"))) {
          const s = parseStmt(st.ir);
          if (s.k === "free" || s.k === "take") { found = true; return; }
          walkE(s.e); walkE(s.lhs);
          if (s.k === "let") c.vars[s.name] = typeOf(s.e, c) === "null" ? "N" : typeOf(s.e, c);
        }
        if (st.body) walk(st.body);
      }
    };
    walk(steps);
    return found;
  }

  // the header line(s) of a function, from its spec
  function signature(name, s, ctx) {
    const L = ctx.lang, g = ctx.generic;
    const T = ctx.T;
    const hints = !ctx.ref && !ctx.harnessFn;
    const n = fnName(name, ctx);
    const struct = (k) => k.slice(k.indexOf(":") + 1);
    const isRefK = (k) => k.startsWith("mut:") || k.startsWith("ref:");
    if (L === "py") {
      const pyType = (k) => (isRefK(k) ? `${struct(k)}${g ? "[T]" : ""}` : { N: `Node${g ? "[T]" : ""} | None`, int: "int", T: g ? "T" : "int" }[k]);
      const ps = s.params.map(([p, k]) => (hints ? `${p}: ${pyType(k)}` : p)).join(", ");
      const ret = hints ? ` -> ${{ void: "None", bool: "bool", int: "int", T: g ? "T" : "int", Tpeek: g ? "T" : "int" }[s.ret]}` : "";
      return [`def ${n}${hints && g ? "[T]" : ""}(${ps})${ret}:`];
    }
    const tg = g ? "<T>" : "";
    if (L === "cpp") {
      const ps = s.params.map(([p, k]) => (k.startsWith("mut:") ? `${struct(k)}${tg}& ${p}` : k.startsWith("ref:") ? `const ${struct(k)}${tg}& ${p}`
        : { N: `Node${tg}* ${p}`, int: `int ${p}`, T: `${T} ${p}` }[k])).join(", ");
      const ret = { void: "void", bool: "bool", int: "int", T, Tpeek: T }[s.ret];
      return [...(g ? ["template <typename T>"] : []), `${ret} ${n}(${ps}) {`];
    }
    if (L === "java") {
      const ps = s.params.map(([p, k]) => (isRefK(k) ? `${struct(k)}${tg} ${p}` : { N: `Node${tg} ${p}`, int: `int ${p}`, T: `${T} ${p}` }[k])).join(", ");
      const ret = { void: "void", bool: "boolean", int: "int", T, Tpeek: T }[s.ret];
      return [`static ${g ? "<T> " : ""}${ret} ${n}(${ps}) {`];
    }
    const ps = s.params.map(([p, k]) => (k.startsWith("mut:") ? `${p}: &mut ${struct(k)}${tg}` : k.startsWith("ref:") ? `${p}: &${struct(k)}${tg}`
      : { N: `${p}: *mut Node${tg}`, int: `${p}: usize`, T: `${p}: ${T}` }[k])).join(", ");
    const bounds = g ? s.rsBounds : [];
    const gen = g ? `<T${bounds.length ? ": " + bounds.join(" + ") : ""}>` : "";
    const ret = { void: "", bool: " -> bool", int: " -> usize", T: ` -> ${T}`, Tpeek: g ? " -> &T" : ` -> ${T}` }[s.ret];
    return [`fn ${n}${gen}(${ps})${ret} {`];
  }

  // A function: { name, steps } -> { header, display (top-level micro-steps incl. header and closing), code }
  function renderFunction(fn, ctx0, opts = {}) {
    const s = ctx0.fns[fn.name];
    const ctx = { ...ctx0, vars: {}, muts: reassigned(fn.steps), harnessFn: !!s.harness, ret: s.ret, owner: s.owner || null };
    s.params.forEach(([p, k]) => { ctx.vars[p] = k.startsWith("mut:") || k.startsWith("ref:") ? `R:${k.slice(4)}` : k; });
    const sp = { ...s, rsBounds: ctx.lang === "rs" && ctx.generic ? rustBounds(fn.steps, { ...ctx, vars: { ...ctx.vars } }) : [] };
    const head = signature(fn.name, sp, ctx);
    const body = toDisplay(fn.steps, ctx);
    const L = ctx.lang;
    const unsafe = L === "rs" && usesPointers(fn.steps, ctx);
    const display = [];
    display.push({ say: opts.headerSay || "", code: head, body: [], close: [], note: null, header: true });
    if (unsafe) {
      display.push({ say: "Open an `unsafe` block for the body: following a raw pointer (`*p`) is unsafe in Rust, so the compiler wants it marked.",
        code: ["unsafe {"], body: [], close: [], note: null, auto: true });
    }
    display.push(...body);
    if (LANGS[L].cLike) display.push({ say: "", code: unsafe ? ["}", "}"] : ["}"], body: [], close: [], note: null, closing: true });
    // full code: header, then the body indented (twice inside Rust's unsafe block)
    const lines = [...head];
    if (unsafe) lines.push("    unsafe {");
    emit(body, unsafe ? 2 : 1, lines);
    if (unsafe) lines.push("    }");
    if (LANGS[L].cLike) lines.push("}");
    return { display, code: lines.join("\n"), header: head };
  }

  // A sequence of statements outside any function (the tests' main), as lines at depth 0.
  function renderLines(steps, ctx) {
    ctx.muts = ctx.muts || reassigned(steps);
    return emit(toDisplay(steps, ctx));
  }

  const indent = (code, d) => code.split("\n").map((l) => (l ? "    ".repeat(d) + l : l)).join("\n");

  const api = { LANGS, camel, parseExpr, parseStmt, typeOf, expr, stmt, toDisplay, emit, renderFunction, renderLines,
    signature, reassigned, rustBounds, indent, dataLit };
  root.DSALang = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
