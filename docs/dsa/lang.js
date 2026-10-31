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
  const TOKEN = /\s*(\d+|"(?:[^"\\]|\\.)*"|[A-Za-z_]\w*|==|!=|<=|>=|[-+*\/%<>().,[\]])/y;
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
      let a = mul();
      while (peek() === "+" || peek() === "-") { const op = next(); a = { k: "bin", op, a, b: mul() }; }
      return a;
    };
    const mul = () => {
      let a = post();
      while (peek() === "*" || peek() === "/" || peek() === "%") { const op = next(); a = { k: "bin", op, a, b: post() }; }
      return a;
    };
    const post = () => {
      let a = prim();
      for (;;) {
        if (peek() === ".") { next(); a = { k: "field", obj: a, name: next() }; }
        else if (peek() === "[") { next(); const idx = or(); expect("]"); a = { k: "index", obj: a, idx }; }
        else return a;
      }
    };
    const prim = () => {
      const s = next();
      if (s === undefined) throw new Error(`IR: unexpected end of ${JSON.stringify(src)}`);
      if (s === "(") { const e = or(); expect(")"); return e; }
      if (s === "[") {   // an array literal (the tests' arrays)
        const items = [];
        if (peek() !== "]") { items.push(or()); while (peek() === ",") { next(); items.push(or()); } }
        expect("]");
        return { k: "array", items };
      }
      if (s === "-" && /^\d/.test(peek() || "")) return { k: "num", v: -Number(next()) };
      if (/^\d/.test(s)) return { k: "num", v: Number(s) };
      if (s[0] === '"') return { k: "str", v: JSON.parse(s) };
      if (s === "null") return { k: "null" };
      if (s === "true" || s === "false") return { k: "bool", v: s === "true" };
      if (s === "default") return { k: "default" };
      if (s === "vec" && peek() === "(") { next(); expect(")"); return { k: "vec" }; }   // an empty growable array
      if ((s === "alloc" || s === "len") && peek() === "(") {   // a fixed block of n slots; an array's length
        next(); const a = or(); expect(")");
        return s === "alloc" ? { k: "alloc", n: a } : { k: "len", a };
      }
      if (peek() === "(") {
        next();
        const args = [];
        if (peek() !== ")") { args.push(or()); while (peek() === ",") { next(); args.push(or()); } }
        expect(")");
        if (s === "allocn") return { k: "allocn", n: args[0] };   // a block of n null pointers (a hash table's slots)
        return s === "new" ? { k: "new", args } : { k: "call", fn: s, args };
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
    if ((m = line.match(/^for (\w+) = (.+) to (.+)$/))) return { k: "for", name: m[1], from: parseExpr(m[2]), to: parseExpr(m[3]) };
    if ((m = line.match(/^(if|elif|while|repeat) (.+)$/))) return { k: m[1], e: parseExpr(m[2]) };
    if (line === "else" || line === "block") return { k: line };
    if ((m = line.match(/^return(?: (.+))?$/))) return { k: "return", e: m[1] ? parseExpr(m[1]) : null };
    if ((m = line.match(/^(assert|free) (.+)$/))) return { k: m[1], e: parseExpr(m[2]) };
    if ((m = line.match(/^(print|println|printb)(?: (.+))?$/))) return { k: m[1], e: m[2] ? parseExpr(m[2]) : null };
    if ((m = line.match(/^make (\w+) (\w+)$/))) return { k: "make", struct: m[1], name: m[2] };
    if ((m = line.match(/^push (\w+), (.+)$/))) return { k: "push", name: m[1], e: parseExpr(m[2]) };
    if ((m = line.match(/^swap (\w+), (.+), (.+)$/))) return { k: "swap", arr: m[1], i: parseExpr(m[2]), j: parseExpr(m[3]) };
    if ((m = line.match(/^(.+?) (\+=|-=|=) (.+)$/))) return { k: "assign", lhs: parseExpr(m[1]), op: m[2], e: parseExpr(m[3]) };
    return { k: "expr", e: parseExpr(line) };
  }

  // ---------------------------------------------------------------- types
  const NODE_FIELDS = { data: "T", next: "N", prev: "N" };
  // the node type: a tutorial can name its own (a hash table's Entry with key, value and next)
  const nodeOf = (ctx) => ctx.node || { name: "Node", fields: NODE_FIELDS };
  const isStruct = (t) => typeof t === "string" && (t.startsWith("R:") || t.startsWith("V:"));

  function typeOf(e, ctx) {
    switch (e.k) {
      case "num": return "int";
      case "array": return "ArrV";
      case "vec": return "Vec";
      case "alloc": return "ArrV";
      case "allocn": return "NArrV";
      case "len": return "int";
      case "index": {
        const ot = typeOf(e.obj, ctx);
        if (ot === "NArrV") return "N";
        if (!["Arr", "ArrV", "Vec"].includes(ot)) throw new Error(`IR: can't index a ${ot}`);
        return "T";
      }
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
        const t = ot === "N" ? nodeOf(ctx).fields[e.name] : isStruct(ot) ? ((ctx.structs[ot.slice(2)] || {}).fields || {})[e.name] : null;
        if (!t) throw new Error(`IR: no field ${e.name} on ${ot}`);
        return t;
      }
      case "call": { const r = spec(e.fn, ctx).ret; return r === "Tpeek" ? "T" : r === "index" ? "index" : r; }
      case "bin": return ["+", "-", "*", "/", "%"].includes(e.op) ? "int" : "bool";
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
  const PREC = { or: 1, and: 2, "==": 4, "!=": 4, "<": 4, ">": 4, "<=": 4, ">=": 4, "+": 5, "-": 5, "*": 6, "/": 6, "%": 6 };

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
    if (ctx.strT) return { py: '""', cpp: "std::string()", java: '""', rs: "String::new()" }[ctx.lang];
    if (!ctx.generic) return "0";
    return { py: "None", cpp: `${ctx.T}{}`, java: "null", rs: `${ctx.T}::default()` }[ctx.lang];
  }

  function newNode(args, ctx) {
    const a = args.map((x) => expr(x, ctx)).join(", ");
    const g = ctx.generic, N = nodeOf(ctx).name;
    return { py: `${N}(${a})`, cpp: g ? `new ${N}<${ctx.T}>(${a})` : `new ${N}(${a})`,
      java: g ? `new ${N}<>(${a})` : `new ${N}(${a})`, rs: `${N}::new(${a})` }[ctx.lang];
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
      case "new": return newNode(e.args, ctx);
      case "id": return ctx.derefs && ctx.derefs.has(e.name) ? `*${e.name}` : e.name;   // (Rust: an element passed as &T)
      case "index": {
        if (L === "java" && typeOf(e.obj, ctx) === "Vec") return `${expr(e.obj, ctx, 9)}.get(${expr(e.idx, ctx)})`;   // an ArrayList
        if (L === "rs" && typeOf(e.idx, ctx) === "index") return `${expr(e.obj, ctx, 9)}[${expr(e.idx, ctx, 9)} as usize]`;
        return `${expr(e.obj, ctx, 9)}[${expr(e.idx, ctx)}]`;
      }
      case "vec": return { py: "[]", cpp: `std::vector<${ctx.T}>{}`, java: "new ArrayList<>()", rs: "Vec::new()" }[L];
      case "alloc": {   // slots hold a placeholder until written
        const n = expr(e.n, ctx), n9 = expr(e.n, ctx, 9);
        const param = ctx.T === "T";   // (inside a generic function, as opposed to the tests' concrete type)
        return { py: `[${ctx.generic ? "None" : "0"}] * ${n9}`, cpp: `std::vector<${ctx.T}>(${n})`,
          java: param ? `(T[]) new Object[${n}]` : `new ${ctx.T}[${n}]`, rs: `vec![${ctx.generic ? `${ctx.T}::default()` : "0"}; ${n}]` }[L];
      }
      case "allocn": {
        const n = expr(e.n, ctx), N = nodeOf(ctx).name;
        return { py: `[None] * ${expr(e.n, ctx, 9)}`, cpp: `std::vector<${N}*>(${n}, nullptr)`, java: `new ${N}[${n}]`,
          rs: `vec![${ctx.qual ? "std::ptr" : "ptr"}::null_mut(); ${n}]` }[L];
      }
      case "len": {
        const a = expr(e.a, ctx, 9);
        return { py: `len(${expr(e.a, ctx)})`, cpp: `(int) ${a}.size()`, java: `${a}.length`, rs: `${a}.len()` }[L];
      }
      case "array": {
        const items = e.items.map((x) => expr(x, ctx)).join(", ");
        return { py: `[${items}]`, cpp: `std::vector<${ctx.T}>{${items}}`, java: `new ${ctx.T}[]{${items}}`, rs: `vec![${items}]` }[L];
      }
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
        // Java orders generic elements with compareTo
        if (p === 4 && L === "java" && ctx.generic && typeOf(e.a, ctx) === "T") return wrap(`${expr(e.a, ctx, 9)}.compareTo(${expr(e.b, ctx)}) ${e.op} 0`, 4);
        const op = e.op === "/" && L === "py" ? "//" : e.op;   // integer division
        if (p === 4) {
          const [a, b] = rsIndexSides(e, typeOf(e.a, ctx), typeOf(e.b, ctx), expr(e.a, ctx, p), expr(e.b, ctx, p + 1), ctx);
          return wrap(`${a} ${op} ${b}`, p);
        }
        return wrap(`${expr(e.a, ctx, p)} ${op} ${expr(e.b, ctx, p + 1)}`, p);
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
    let a = expr(e.a, ctx, 5), b = expr(e.b, ctx, 5);
    if (ta === "N" && L === "py") return `${a} ${eq ? "is" : "is not"} ${b}`;
    if (ta === "T" && L === "java" && (ctx.generic || ctx.strT)) return `${eq ? "" : "!"}${expr(e.a, ctx, 9)}.equals(${b})`;
    [a, b] = rsIndexSides(e, ta, tb, a, b, ctx);
    return `${a} ${e.op} ${b}`;
  }

  // Rust: an index (isize, maybe -1) compared with a position (usize) needs the position cast
  function rsIndexSides(e, ta, tb, a, b, ctx) {
    if (ctx.lang !== "rs") return [a, b];
    if (ta === "index" && tb === "int" && e.b.k !== "num") return [a, `(${b} as isize)`];
    if (tb === "index" && ta === "int" && e.a.k !== "num") return [`(${a} as isize)`, b];
    return [a, b];
  }

  // Rust can't move a generic element out of an array: copying one (into a variable, another array or a slot) clones it
  function copyOut(e, rendered, ctx) {
    const copies = e && (e.k === "index" || (e.k === "field" && typeOf(e, ctx) === "T"));   // (an array slot, or a node's element)
    return ctx.lang === "rs" && ctx.generic && copies ? `${rendered}.clone()` : rendered;
  }

  // a call argument: Rust borrows a structure held by value (a local, or a field like Q.inbox), an array, and an
  // element passed by reference (&T)
  function arg(a, kind, ctx) {
    if (ctx.lang === "rs" && kind === "Tref") {
      if (a.k === "id" && ctx.derefs && ctx.derefs.has(a.name)) return a.name;
      return `&${expr(a, ctx, 9)}`;
    }
    const s = expr(a, ctx);
    if (ctx.lang !== "rs") return s;
    const t = String(typeOf(a, ctx));
    if (t.startsWith("V:")) return kind.startsWith("mut:") ? `&mut ${s}` : `&${s}`;
    if (t === "ArrV") return kind === "ArrMut" ? `&mut ${s}` : `&${s}`;
    return s;
  }

  // ---------------------------------------------------------------- type names
  function typeName(t, ctx) {
    const g = ctx.generic ? `<${ctx.T}>` : "";
    if (t === "N") return ctx.lang === "cpp" ? `${nodeOf(ctx).name}${g}*` : `${nodeOf(ctx).name}${g}`;
    if (t === "NArrV") return ctx.lang === "cpp" ? `std::vector<${nodeOf(ctx).name}*>` : `${nodeOf(ctx).name}[]`;
    if (t === "ArrV") return ctx.lang === "cpp" ? `std::vector<${ctx.T}>` : `${ctx.T}[]`;
    if (t === "Vec") {
      const boxed = { int: "Integer" }[ctx.T] || ctx.T;
      return ctx.lang === "cpp" ? `std::vector<${ctx.T}>` : `${ctx.qual ? "java.util." : ""}ArrayList<${boxed}>`;
    }
    if (t === "index") return "int";
    if (t === "T") return ctx.T;
    if (t === "int" || t === "V") return "int";
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
        if (L === "rs") {
          const mut = ctx.muts.has(s.name) || vt === "Vec" || vt === "ArrV" || vt === "NArrV" ? "mut " : "";   // (arrays may be changed through &mut)
          const ann = vt === "ArrV" || vt === "Vec" ? `: Vec<${ctx.T}>` : vt === "NArrV" ? `: Vec<*mut ${nodeOf(ctx).name}>` : "";
          return { code: [`let ${mut}${s.name}${ann} = ${copyOut(s.e, val, ctx)};`] };
        }
        if (vt === "Vec" && L === "cpp") return { code: [`${typeName(vt, ctx)} ${s.name};`] };
        if (vt === "Vec" && L === "java") return { code: [`${typeName(vt, ctx)} ${s.name} = new ${ctx.qual ? "java.util." : ""}ArrayList<>();`] };
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
        return { code: [`${lhs} ${s.op} ${copyOut(s.e, expr(s.e, ctx), ctx)}${semi}`] };
      }
      case "push": {   // append to a growable array
        const v = copyOut(s.e, expr(s.e, ctx), ctx);
        const b = s.name;
        return { code: [{ py: `${b}.append(${v})`, cpp: `${b}.push_back(${v});`, java: `${b}.add(${v});`, rs: `${b}.push(${v});` }[L]] };
      }
      case "swap": {   // the checker's own swaps (the tutorials teach swap as a function)
        const [x, i, j] = [s.arr, expr(s.i, ctx), expr(s.j, ctx)];
        const vec = ctx.vars[x] === "Vec";
        return { code: [{ py: `${x}[${i}], ${x}[${j}] = ${x}[${j}], ${x}[${i}]`, cpp: `std::swap(${x}[${i}], ${x}[${j}]);`,
          java: vec ? `java.util.Collections.swap(${x}, ${i}, ${j});` : `{ ${ctx.T} t = ${x}[${i}]; ${x}[${i}] = ${x}[${j}]; ${x}[${j}] = t; }`,
          rs: `${x}.swap(${i}, ${j});` }[L]] };
      }
      case "if": return { code: [open(`if ${cond(s.e)}`)], close: end };
      case "elif": return { code: [L === "py" ? `elif ${expr(s.e, ctx)}:` : `} else if ${cond(s.e)} {`], close: end };
      case "else": return { code: [L === "py" ? "else:" : "} else {"], close: end };
      case "while": return { code: [open(`while ${cond(s.e)}`)], close: end };
      case "for": {   // for i = lo to hi: i runs over [lo, hi)
        ctx.vars[s.name] = "int";
        const a = expr(s.from, ctx), b = expr(s.to, ctx);
        const head = { py: `for ${s.name} in range(${a}, ${b}):`, cpp: `for (int ${s.name} = ${a}; ${s.name} < ${b}; ${s.name}++) {`,
          java: `for (int ${s.name} = ${a}; ${s.name} < ${b}; ${s.name}++) {`, rs: `for ${s.name} in ${a}..${b} {` }[L];
        return { code: [head], close: end };
      }
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
        const borrow = L === "rs" && ctx.generic && ctx.ret === "Tpeek" && (s.e.k === "field" || s.e.k === "index") ? "&" : "";
        // Rust: an index (or -1) is an isize, the positions themselves usize
        if (L === "rs" && ctx.ret === "index" && s.e.k !== "num" && typeOf(s.e, ctx) === "int") return { code: [`return ${expr(s.e, ctx, 9)} as isize;`] };
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
      if (e.k === "alloc") { need.add("Default"); need.add("Clone"); }   // (vec![T::default(); n])
      if (e.k === "bin" && (e.op === "==" || e.op === "!=") && typeOf(e.a, c) === "T") need.add("PartialEq");
      if (e.k === "bin" && ["<", ">", "<=", ">="].includes(e.op) && typeOf(e.a, c) === "T") need.add("PartialOrd");
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
          if (s.k === "for") c.vars[s.name] = "int";
          if ((s.k === "assign" || s.k === "push" || s.k === "let") && s.e && (s.e.k === "index" || (s.e.k === "field" && typeOf(s.e, c) === "T"))) need.add("Clone");
        }
        if (st.body) walk(st.body);
      }
    };
    walk(steps);
    (ctx.ownBounds || []).forEach((b) => need.add(b));
    if (need.has("PartialOrd")) need.delete("PartialEq");   // (PartialOrd includes it)
    return ["PartialEq", "PartialOrd", "Clone", "std::fmt::Display", "Default"].filter((b) => need.has(b));
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
    const NN = nodeOf(ctx).name;
    if (L === "py") {
      const Tp = g ? "T" : T;
      const pyType = (k) => (isRefK(k) ? `${struct(k)}${g ? "[T]" : ""}` : { N: `${NN}${g ? "[T]" : ""} | None`, int: "int", V: "int", T: Tp,
        Tref: Tp, Arr: `list[${Tp}]`, ArrMut: `list[${Tp}]` }[k]);
      const ps = s.params.map(([p, k]) => (hints ? `${p}: ${pyType(k)}` : p)).join(", ");
      const ret = hints ? ` -> ${{ void: "None", bool: "bool", int: "int", V: "int", index: "int", T: Tp, Tpeek: Tp, N: pyType("N") }[s.ret]}` : "";
      return [`def ${n}${hints && g ? "[T]" : ""}(${ps})${ret}:`];
    }
    const tg = g ? "<T>" : "";
    if (L === "cpp") {
      const ps = s.params.map(([p, k]) => (k.startsWith("mut:") ? `${struct(k)}${tg}& ${p}` : k.startsWith("ref:") ? `const ${struct(k)}${tg}& ${p}`
        : { N: `${NN}${tg}* ${p}`, int: `int ${p}`, V: `int ${p}`, T: `${T} ${p}`, Tref: g ? `const T& ${p}` : `${T} ${p}`, Arr: `const std::vector<${T}>& ${p}`, ArrMut: `std::vector<${T}>& ${p}` }[k])).join(", ");
      const ret = { void: "void", bool: "bool", int: "int", V: "int", index: "int", T, Tpeek: T, N: `${NN}${tg}*` }[s.ret];
      return [...(g ? ["template <typename T>"] : []), `${ret} ${n}(${ps}) {`];
    }
    if (L === "java") {
      const ps = s.params.map(([p, k]) => (isRefK(k) ? `${struct(k)}${tg} ${p}` : { N: `${NN}${tg} ${p}`, int: `int ${p}`, V: `int ${p}`, T: `${T} ${p}`, Tref: `${T} ${p}`, Arr: `${T}[] ${p}`, ArrMut: `${T}[] ${p}` }[k])).join(", ");
      const ret = { void: "void", bool: "boolean", int: "int", V: "int", index: "int", T, Tpeek: T, N: `${NN}${tg}` }[s.ret];
      const gen = !g ? "" : (s.rsBounds || []).includes("PartialOrd") ? "<T extends Comparable<T>> " : "<T> ";
      return [`static ${gen}${ret} ${n}(${ps}) {`];
    }
    const ps = s.params.map(([p, k]) => (k.startsWith("mut:") ? `${p}: &mut ${struct(k)}${tg}` : k.startsWith("ref:") ? `${p}: &${struct(k)}${tg}`
      : { N: `${p}: *mut ${NN}${tg}`, int: `${p}: usize`, V: `${p}: i32`, T: `${p}: ${T}`, Tref: `${p}: &${T}`, Arr: `${p}: &[${T}]`, ArrMut: `${p}: &mut [${T}]` }[k])).join(", ");
    const bounds = g ? s.rsBounds : [];
    const gen = g ? `<T${bounds.length ? ": " + bounds.join(" + ") : ""}>` : "";
    const ret = { void: "", bool: " -> bool", int: " -> usize", V: " -> i32", index: " -> isize", T: ` -> ${T}`, Tpeek: g ? " -> &T" : ` -> ${T}`, N: ` -> *mut ${NN}${tg}` }[s.ret];
    return [`fn ${n}${gen}(${ps})${ret} {`];
  }

  // A function: { name, steps } -> { header, display (top-level micro-steps incl. header and closing), code }
  function renderFunction(fn, ctx0, opts = {}) {
    const s = ctx0.fns[fn.name];
    const ctx = { ...ctx0, vars: {}, muts: reassigned(fn.steps), harnessFn: !!s.harness, ret: s.ret, owner: s.owner || null };
    s.params.forEach(([p, k]) => { ctx.vars[p] = k.startsWith("mut:") || k.startsWith("ref:") ? `R:${k.slice(4)}` : k === "Tref" ? "T" : k === "ArrMut" ? "Arr" : k; });
    // Rust reads an element passed by reference as *x
    if (ctx.lang === "rs") ctx.derefs = new Set(s.params.filter(([, k]) => k === "Tref").map(([p]) => p));
    // what T must support: Rust's trait bounds, and Java's Comparable for ordering
    const needsBounds = ctx.generic && (ctx.lang === "rs" || ctx.lang === "java");
    const sp = { ...s, rsBounds: needsBounds ? rustBounds(fn.steps, { ...ctx, vars: { ...ctx.vars }, ownBounds: s.bounds }) : [] };
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
