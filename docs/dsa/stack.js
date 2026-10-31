// 13.05 Stacks: a stack on a singly linked list (push and pop at the head), in Python, C++, Java or Rust.
(function (root) {
  "use strict";
  const DL = root.DSALang || (typeof require !== "undefined" ? require("./lang.js") : null);
  const C = root.DSACommon || (typeof require !== "undefined" ? require("./common.js") : null);
  const { S, fname, VALUES } = C;

  const OPTIONS = [
    { id: "lang", label: "Language", choices: [["py", "Python"], ["cpp", "C++"], ["java", "Java"], ["rs", "Rust"]] },
    { id: "type", label: "Elements", choices: [["int", "int"], ["generic", "generic (T)"]],
      help: { int: "Every element is an integer.", generic: "Any element type: a type parameter T (templates in C++, generics in Java and Rust, type parameters in Python)." } },
    { id: "impl", label: "Built on", choices: [["list", "a linked list"]],
      help: { list: "Push and pop at the front of a singly linked list: the top of the stack is its first node." } },
  ];
  const DEFAULTS = { lang: "py", type: "int", impl: "list" };
  const describe = (o) => `stack of ${o.type === "int" ? "int" : "generic"} elements on a linked list, in ${DL.LANGS[o.lang].name}`;

  function complexity() {
    return [
      { id: "space", label: "Space", big: "O(n)", note: "n nodes × (element + 1 pointer)" },
      { id: "push", label: "Push", big: "O(1)", note: "a new node on top" },
      { id: "pop", label: "Pop", big: "O(1)", note: "the top is one pointer away" },
      { id: "peek", label: "Peek", big: "O(1)", note: "read the top's element" },
      { id: "size", label: "Size", big: "O(1)", note: "kept in `S.size`" },
      { id: "empty", label: "Is empty", big: "O(1)", note: "`S.size == 0`" },
      { id: "search", label: "Search", big: "O(n)", note: "not a stack operation: walk down from the top", extra: true },
      { id: "bottom", label: "Bottom element", big: "O(n)", note: "walk all the way down", extra: true },
      { id: "is_stack", label: "Contract is_stack", big: "O(n)", note: "walks the stack, so contracts make every call O(n) while testing" },
    ];
  }

  function diagram(o) {
    const [a, b, c] = o.type === "int" ? ["30", "20", "10"] : ["cat", "bee", "ant"];
    return {
      rows: [[{ box: a, tags: ["S.top"] }, { arrow: "→" }, { box: b, tags: [] }, { arrow: "→" }, { box: c, tags: [] }, { arrow: "→" }, { text: "null" }]],
      caption: "The stack is the segment [S.top, null): the top is the first node, and push and pop both happen there. Empty = S.top is null.",
    };
  }

  // ---------------------------------------------------------------- specs
  const M = "mut:Stack", R = "ref:Stack", O = "Stack";
  const FNS = {
    is_segment: { params: [["start", "N"], ["end", "N"], ["n", "int"]], ret: "bool", owner: O, contract: true },
    is_stack: { params: [["S", R]], ret: "bool", owner: O, contract: true },
    stack_empty: { params: [["S", R]], ret: "bool", owner: O },
    push: { params: [["S", M], ["x", "T"]], ret: "void", owner: O },
    pop: { params: [["S", M]], ret: "T", owner: O },
    peek: { params: [["S", R]], ret: "Tpeek", owner: O },
    stack_size: { params: [["S", R]], ret: "int", owner: O },
    show: { params: [["S", R]], ret: "void", harness: true, bounds: ["std::fmt::Display"] },
  };
  const STRUCTS = { Stack: { fields: { top: "N", size: "int" } } };

  // ---------------------------------------------------------------- the functions
  const REQ = S("**Requires**: the stack is valid when the function starts.", "assert is_stack(S)");
  const ENS = S("**Ensures**: the stack is still valid when it returns.", "assert is_stack(S)");
  const NONEMPTY = S("**Requires**: the stack is not empty.", "assert not stack_empty(S)");

  // the stack's segment ends at null (shared with 13.06's two-stack queue)
  function isSegment() {
    return [
      S("Start at `start`.", "let p = start"),
      S("Take exactly `n` steps. (Loop invariant: after k rounds, `p` is k nodes past `start`, and none of them was `end`.)", "repeat n", [
        S("Hitting {null} or `end` before `n` steps means the segment is shorter than `n`.", "if p == null or p == end", [S("", "return false")]),
        S("Step to the next node.", "p = p.next"),
      ]),
      S("After `n` steps we must stand exactly on `end`: then [`start`, `end`) holds exactly `n` nodes.", "return p == end"),
    ];
  }
  const isStack = () => [S("The stack is the segment [`S.top`, {null}): exactly `S.size` nodes, then {null}.", "return is_segment(S.top, null, S.size)")];
  const stackEmpty = () => [S("Empty exactly when the size is 0.", "return S.size == 0")];
  const push = () => [REQ,
    S("Make a new node `n` holding `x`.", "let n = new(x)"),
    S("`n` goes on top of the old top ({null} if the stack is empty).", "n.next = S.top"),
    S("`n` is the new top.", "S.top = n"),
    S("One more element.", "S.size += 1"), ENS];
  const pop = () => [REQ, NONEMPTY,
    S("`n` is the top node.", "let n = S.top"),
    S("The node under it is the new top ({null} if `n` was the last one).", "S.top = n.next"),
    S("One element fewer.", "S.size -= 1"),
    S("Save the element and free the node.", "let x = take n"), ENS,
    S("Return the element.", "return x")];
  const peek = () => [REQ, NONEMPTY,
    S((c) => (c.lang === "rs" && c.generic
      ? "Return the top element without removing it. Rust can't move it out of the node, so `peek` returns a reference `&T`."
      : "Return the top element without removing it."), "return S.top.data")];
  const stackSize = () => [S("The size is kept in the stack.", "return S.size")];

  // ---------------------------------------------------------------- declarations, per language
  const FIELD_SAY = {
    top: "`top`: the top node ({null} when the stack is empty). It's the first node of a linked list.",
    size: "`size`: the number of elements, so the size is O(1).",
  };
  function stackDecl(c) {
    const g = c.generic, L = c.lang;
    if (L === "py") {
      return [{ say: g ? "Declare the class `Stack[T]`." : "Declare the class `Stack`.", code: [g ? "class Stack[T]:" : "class Stack:"], body: [
        { say: "The constructor makes an empty stack.", code: ["def __init__(self) -> None:"], body: [
          { say: FIELD_SAY.top, code: [`self.top: ${g ? "Node[T]" : "Node"} | None = None`] },
          { say: FIELD_SAY.size, code: ["self.size = 0"] }] }] }];
    }
    if (L === "cpp") {
      const P = g ? "Node<T>*" : "Node*";
      return [{ say: (g ? "Declare the stack struct, a template over `T`." : "Declare the stack struct `Stack`.") + " No constructor needed: the defaults make an empty stack.",
        code: g ? ["template <typename T>", "struct Stack {"] : ["struct Stack {"], close: ["};"], body: [
          { say: FIELD_SAY.top, code: [`${P} top = nullptr;`] }, { say: FIELD_SAY.size, code: ["int size = 0;"] }] }];
    }
    if (L === "java") {
      return [{ say: (g ? "Declare the class `Stack<T>`." : "Declare the class `Stack`.") + " Fields start as `null` and 0: an empty stack. Every function from here on goes inside this class, as a `static` method that takes the stack `S`.",
        code: [g ? "class Stack<T> {" : "class Stack {"], close: ["}"], deferClose: true, cls: "Stack", body: [
          { say: FIELD_SAY.top, code: [`${g ? "Node<T>" : "Node"} top;`] }, { say: FIELD_SAY.size, code: ["int size;"] }] }];
    }
    const N = g ? "Node<T>" : "Node";
    return [
      { say: g ? "Declare the struct `Stack<T>`." : "Declare the struct `Stack`.", code: [g ? "struct Stack<T> {" : "struct Stack {"], close: ["}"], body: [
        { say: FIELD_SAY.top, code: [`top: *mut ${N},`] }, { say: FIELD_SAY.size, code: ["size: usize,"] }] },
      { say: "A constructor `Stack::new()` that makes an empty stack.", code: [g ? "impl<T> Stack<T> {" : "impl Stack {"], close: ["}"], body: [
        { say: "", code: ["fn new() -> Self {"], close: ["}"], body: [{ say: "A null top, size 0.", code: ["Stack { top: ptr::null_mut(), size: 0 }"] }] }] },
    ];
  }

  const RUST_WHY = "A stack alone could own its nodes with `Option<Box<…>>`, but a queue (13.06) points at its last node from two places, which needs raw pointers, so these tutorials use them throughout. `std::collections::LinkedList` is built the same way.";

  function steps(o) {
    const fn = (name, body, headerSay) => ({ name, steps: body(), headerSay });
    const on = C.contractsOn(o);
    return [
      { id: "node", title: "The node",
        goal: "The stack is a linked list whose first node is the top. Each node holds one element (`data`) and a pointer to the node below it. A pointer that points nowhere is null.",
        decl: (c) => C.nodeSteps(c, { varName: "S", what: "stack", rustWhy: RUST_WHY }), tests: ["node"] },
      { id: "stack", title: on ? "The stack and its contracts" : "The stack",
        goal: !on ? "The stack `S` keeps its `top` node and its `size`. Write the struct, its constructor, and {fn:stack_empty(S)}." :
          "The stack `S` keeps its `top` node and its `size`. Before any operation, write the **contracts**: {fn:is_segment(start, end, n)} " +
          "(exactly `n` nodes from `start` up to, not including, `end`) and {fn:is_stack(S)}, which is true exactly when [`S.top`, null) holds `S.size` nodes. " +
          "Every operation checks {fn:is_stack(S)} when it starts and before it returns.",
        decl: (c) => stackDecl(c),
        fns: [fn("is_segment", isSegment, "Write {fn:is_segment}: does [`start`, `end`) hold exactly `n` nodes?"),
          fn("is_stack", isStack, "Write {fn:is_stack}, the representation invariant."),
          fn("stack_empty", stackEmpty, "Write {fn:stack_empty}.")],
        contractNote: true, tests: ["node", "stack"] },
      { id: "push", title: "Push",
        goal: "{fn:push(S, x)} puts `x` on top: a new first node. O(1).",
        fns: [fn("push", push, "Write the header: {fn:push} takes the stack and the new element.")], tests: ["node", "stack", "push"] },
      { id: "pop", title: "Pop",
        goal: "{fn:pop(S)} removes the top element and returns it. It **requires** a non-empty stack. O(1).",
        fns: [fn("pop", pop, "Write the header: {fn:pop} returns the removed element.")], tests: ["node", "stack", "push", "pop"] },
      { id: "peek_size", title: "Peek and size",
        goal: "{fn:peek(S)} returns the top element without removing it (it also requires a non-empty stack), and {fn:stack_size(S)} returns how many elements there are. Both O(1).",
        fns: [fn("peek", peek, "Write {fn:peek}."), fn("stack_size", stackSize, "Write {fn:stack_size}.")],
        tests: ["node", "stack", "push", "pop", "peek_size"] },
    ];
  }

  // ---------------------------------------------------------------- tests
  const BREAKS = [["S.size += 1", "S.size is one too big"], ["S.size -= 1", "S.size is one too small"],
    ["S.top.next.next.next = S.top", "the bottom node points back to the top"], ["S.top = null", "S.top is null"]];

  function showFn() {
    return [S("", 'print "["'), S("", "let p = S.top"), S("", "let count = 0"),
      S("", "while p != null and count < S.size + 2", [S("", "if count > 0", [S("", 'print ", "')]), S("", "print p.data"), S("", "p = p.next"), S("", "count += 1")]),
      S("", 'print "] size="'), S("", "print S.size"),
      S("", "if not is_segment(S.top, null, S.size)", [S("", 'print " BROKEN: following next from S.top does not reach null after exactly S.size nodes (check size and next pointers)"'),
        S("", "println"), S("", "return")]),
      S("", "println")];
  }

  function handlers(o) {
    const nm = (n) => fname(n, o.lang);
    const { lit, say } = C;
    const show = (A) => `[${A.join(", ")}] size=${A.length}`;
    return {
      node: C.ops.node(false),
      new: (x, st) => { st.A = []; return { ir: [S("", "make Stack S"), say("new stack: "), S("", "show(S)")], out: `new stack: ${show(st.A)}` }; },
      fill: (xs, st) => { xs.forEach((y) => st.A.unshift(y)); return { ir: xs.map((y) => S("", `ref_push(S, ${lit(y)})`)) }; },
      push: (x, st) => { st.A.unshift(x); return { ir: [say(`push(S, ${x}): `), S("", `push(S, ${lit(x)})`), S("", "show(S)")], out: `push(S, ${x}): ${show(st.A)}` }; },
      pop: (x, st) => { const y = st.A.shift(); return { ir: [say("pop(S) = "), S("", "print pop(S)"), say(": "), S("", "show(S)")], out: `pop(S) = ${y}: ${show(st.A)}` }; },
      peek: (x, st) => C.ops.value(o, "peek(S)", "peek(S)", st.A[0]),
      stack_size: (x, st) => C.ops.value(o, "stack_size(S)", `${nm("stack_size")}(S)`, st.A.length),
      stack_empty: (x, st) => C.ops.bool(o, "stack_empty(S)", `${nm("stack_empty")}(S)`, st.A.length === 0),
      is_stack: () => C.ops.bool(o, "is_stack(S)", `${nm("is_stack")}(S)`, true),
      break: (k) => ({ ir: [S("", BREAKS[k][0]), say(`after ${BREAKS[k][1]}: ${nm("is_stack")}(S) = `), S("", "printb is_stack(S)"), S("", "println")],
        out: `after ${BREAKS[k][1]}: ${nm("is_stack")}(S) = false` }),
      segment: (d) => {
        const n = `S.size${d === 0 ? "" : d > 0 ? ` + ${d}` : ` - ${-d}`}`;
        return C.ops.bool(o, `is_segment(S.top, null, ${n})`, `${nm("is_segment")}(S.top, null, ${n})`, d === 0);
      },
    };
  }

  function testGroups(names, o, seed) {
    const [a, b, c, d, e, f] = VALUES[o.type];
    const G = {
      node: [[["node", a]], [["node", f]]],
      stack: [
        [["new"], ["stack_empty"], ["is_stack"]],
        [["new"], ["fill", [a, b, c]], ["stack_empty"], ["is_stack"], ["segment", 0], ["segment", -1], ["segment", 1], ["segment", 2]],
        [["new"], ["fill", [d]], ["is_stack"], ["segment", 0]],
        ...BREAKS.map((_, k) => [["new"], ["fill", [a, b, c]], ["break", k]]),
      ],
      push: [[["new"], ["push", a], ["push", b], ["push", c], ["is_stack"], ["stack_empty"]]],
      pop: [[["new"], ["push", a], ["push", b], ["pop"], ["push", c], ["pop"], ["pop"], ["stack_empty"], ["push", d], ["pop"], ["stack_empty"]]],
      peek_size: [[["new"], ["stack_size"], ["push", a], ["peek"], ["stack_size"], ["push", b], ["peek"], ["stack_size"], ["pop"], ["peek"], ["stack_size"]]],
    };
    const groups = names.flatMap((n) => G[n]);
    if (seed) {
      const r = C.mulberry32(seed);
      const vals = VALUES[o.type];
      const ops = [["new"]];
      let n = 0;
      for (let i = 0; i < 60; i++) {
        const x = r();
        if (x < 0.4) { ops.push(["push", vals[Math.floor(r() * vals.length)]]); n++; }
        else if (x < 0.65 && n) { ops.push(["pop"]); n--; }
        else if (x < 0.8 && n) ops.push(["peek"]);
        else if (x < 0.9) ops.push(["stack_size"]);
        else if (x < 0.95) ops.push(["stack_empty"]);
        else ops.push(["is_stack"]);
      }
      groups.push(ops);
    }
    return groups;
  }

  // the checker's own helpers, built on the user's Node and Stack (once the stack exists)
  const refFns = (o, names) => (names.includes("stack")
    ? [{ name: "is_segment", steps: isSegment() }, { name: "push", steps: push() }, { name: "show", steps: showFn() }] : []);

  function problem(o) {
    const L = o.lang;
    const sig = C.headers(DEF, o);
    const what = {
      is_segment: "true if following next from start reaches end after exactly n nodes",
      is_stack: "the representation invariant: [top, null) holds exactly size nodes",
      stack_empty: "true if the stack has no elements",
      push: "put x on top, O(1)",
      pop: "remove and return the top element (requires a non-empty stack), O(1)",
      peek: "the top element, without removing it (requires a non-empty stack), O(1)",
      stack_size: "the number of elements, O(1)",
    };
    return [
      `Build a **${describe(o)}** from a blank file, with no hints: everything the tutorial walked you through.`, "",
      `**Node.** Fields \`data\` and \`next\`; ${L === "rs" ? "`Node::new(data)` returns a raw pointer to a new heap node" : "a constructor taking the element"}, with a null \`next\`.`, "",
      `**Stack.** Fields \`top\` and \`size\`. ${{ py: "`Stack()`", cpp: "`Stack S;`", java: "`new Stack()`", rs: "`Stack::new()`" }[L]} makes an empty stack.`, "",
      "**Functions** (exact names and signatures):", "",
      ...Object.keys(what).map((n) => `* \`${sig(n)}\`: ${what[n]}`), "",
      `Check \`${fname("is_stack", L)}(S)\` at the start and end of every operation. ${L === "java" ? "Put the functions inside `class Stack` as static methods. " : ""}` +
        "You don't need a `main`: the checker adds its own (yours, if any, is ignored) and runs every test from the tutorial plus a random sequence of operations.",
    ].join("\n");
  }

  const DEF = {
    id: "stacks", num: "13.05", title: "Stacks", heading: "Build a stack, step by step", diagramLabel: "Your stack",
    OPTIONS, DEFAULTS, FNS, STRUCTS, describe, complexity, diagram, steps, handlers, testGroups, refFns, problem,
    CONTRACT_OPS: ["is_stack", "segment", "break"],
    _internals: { isSegment, isStack, stackEmpty, push, pop, peek, stackSize, stackDecl, showFn, BREAKS, RUST_WHY },
  };
  const tutorial = C.make(DEF);
  if (typeof module !== "undefined" && module.exports) module.exports = tutorial;
})(typeof window !== "undefined" ? window : globalThis);
