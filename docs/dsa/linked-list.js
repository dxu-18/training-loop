// 13.01 Linked lists: the step-by-step tutorial (options, complexity, steps, tests) for docs/dsa/page.js.
//
// Eight layouts: {NULL-terminated, dummy node} x {singly, doubly} x {linear, circular}, each in four languages
// with int or generic elements. Every function is written once as micro-steps in DSALang's statement language.
(function (root) {
  "use strict";
  const DL = root.DSALang || (typeof require !== "undefined" ? require("./lang.js") : null);
  const { parseExpr, expr } = DL;

  // ---------------------------------------------------------------- options
  const OPTIONS = [
    { id: "lang", label: "Language", choices: [["py", "Python"], ["cpp", "C++"], ["java", "Java"], ["rs", "Rust"]] },
    { id: "type", label: "Elements", choices: [["int", "int"], ["generic", "generic (T)"]],
      help: { int: "Every element is an integer.", generic: "Any element type: a type parameter T (templates in C++, generics in Java and Rust, type parameters in Python)." } },
    { id: "end", label: "End of the list", choices: [["null", "NULL-terminated"], ["dummy", "dummy node"]],
      help: { null: "The list ends where a pointer is null.", dummy: "A dummy node marks the end; it holds no element, so no pointer in the list is ever null." } },
    { id: "links", label: "Links", choices: [["singly", "singly"], ["doubly", "doubly"]],
      help: { singly: "Each node points to the next one.", doubly: "Each node also points back to the previous one." } },
    { id: "shape", label: "Shape", choices: [["linear", "linear"], ["circular", "circular"]],
      help: { linear: "The last node is the end.", circular: "The last node points back around to the front." } },
  ];
  const DEFAULTS = { lang: "py", type: "int", end: "null", links: "singly", shape: "linear" };
  const variant = (o) => ({ dummy: o.end === "dummy", doubly: o.links === "doubly", circular: o.shape === "circular" });
  // which pointers the list header keeps
  function fields(v) {
    if (v.dummy) return v.circular ? ["dummy"] : ["head", "dummy"];
    if (v.circular) return v.doubly ? ["head"] : ["tail"];
    return ["head", "tail"];
  }
  function describe(o) {
    const v = variant(o);
    return `${v.circular ? "circular " : ""}${v.doubly ? "doubly" : "singly"} linked list with ${v.dummy ? "a dummy node" : "NULL at the end"}` +
      `, ${o.type === "int" ? "int elements" : "generic elements"}, in ${DL.LANGS[o.lang].name}`;
  }

  // ---------------------------------------------------------------- complexity (updates as options change)
  function complexity(o) {
    const v = variant(o);
    const backVia = v.dummy ? (v.doubly && v.circular ? "through `L.dummy.prev`" : "write into the dummy, add a new dummy")
      : v.circular ? (v.doubly ? "through `L.head.prev`" : "through `L.tail`") : "through `L.tail`";
    const ptrs = v.doubly ? "2 pointers" : "1 pointer";
    return [
      { id: "space", label: "Space", big: "O(n)", note: `n nodes × (element + ${ptrs})${v.dummy ? " + 1 dummy" : ""}` },
      { id: "insert_front", label: "Insert at start", big: "O(1)", note: "a few pointer changes, no loop" },
      { id: "insert_back", label: "Insert at end", big: "O(1)", note: backVia },
      { id: "delete_head", label: "Delete head", big: "O(1)", note: "the first node is one pointer away" },
      { id: "delete_last", label: "Delete last", big: v.doubly ? "O(1)" : "O(n)", note: v.doubly ? "the last node's prev is the new last" : "must walk to the node before the last", extra: true },
      { id: "unlink", label: "Unlink a node you hold", big: v.doubly ? "O(1)" : "O(n)", note: v.doubly ? "its prev and next are right there" : "must find the node before it", extra: true },
      { id: "delete_node", label: "Delete by value", big: "O(n)", note: "search first, then unlink" },
      { id: "lookup", label: "Lookup", big: "O(n)", note: "check the nodes one by one" },
      { id: "min", label: "Find minimum", big: "O(n)", note: "no order to exploit", extra: true },
      { id: "kth", label: "k-th element", big: "O(k)", note: "no indexing: walk k nodes", extra: true },
      { id: "print_list", label: "Traverse / print", big: "O(n)", note: "visit every node once" },
      { id: "size", label: "Length", big: "O(1)", note: "kept in `L.size`" },
      { id: "is_list", label: "Contract is_list", big: "O(n)", note: "walks the list, so contracts make every call O(n) while testing" },
    ];
  }

  // ---------------------------------------------------------------- diagram
  // A drawing of a three-element list in the chosen layout: [{tag, box, arrow}] cells, rendered by page.js.
  function diagram(o) {
    const v = variant(o);
    const els = o.type === "int" ? ["3", "5", "7"] : ["ant", "bee", "cat"];
    const link = v.doubly ? "⇄" : "→";
    const cells = [];
    const node = (label, tags, cls = "") => cells.push({ box: label, tags, cls });
    const arrow = (a) => cells.push({ arrow: a });
    if (!v.dummy && !v.circular) {
      if (v.doubly) { cells.push({ text: "null" }); arrow("←"); }
      els.forEach((e, i) => { node(e, i === 0 ? ["head"] : i === 2 ? ["tail"] : []); arrow(i < 2 ? link : "→"); });
      cells.push({ text: "null" });
    } else if (!v.dummy) {
      els.forEach((e, i) => { node(e, v.doubly ? (i === 0 ? ["head"] : []) : (i === 2 ? ["tail"] : [])); if (i < 2) arrow(link); });
      cells.push({ loop: v.doubly ? "⇄ back to the first" : "→ back to the first" });
    } else if (!v.circular) {
      if (v.doubly) { cells.push({ text: "null" }); arrow("←"); }
      els.forEach((e, i) => { node(e, i === 0 ? ["head"] : []); arrow(link); });
      node("dummy", ["dummy"], "dummy");
      arrow("→"); cells.push({ text: "null" });
    } else {
      node("dummy", ["dummy"], "dummy"); arrow(link);
      els.forEach((e, i) => { node(e, []); if (i < 2) arrow(link); });
      cells.push({ loop: `${link} back to the dummy` });
    }
    const caption = v.dummy
      ? (v.circular ? "The list is the segment [L.dummy.next, L.dummy): around the circle from the first node up to, not including, the dummy. Empty = the dummy pointing to itself."
        : "The list is the segment [L.head, L.dummy): from the first node up to, not including, the dummy. Empty = L.head is the dummy.")
      : (v.circular ? (v.doubly ? "L.head.prev is the last node, so one pointer reaches both ends. Empty = L.head is null."
        : "L.tail.next is the first node, so one pointer reaches both ends. Empty = L.tail is null.")
        : "The list is the segment [L.head, null). L.tail makes the end reachable in one step. Empty = both are null.");
    return { cells, caption };
  }

  // ---------------------------------------------------------------- language helpers for step text
  const NULLW = { py: "None", cpp: "nullptr", java: "null", rs: "ptr::null_mut()" };
  function textCtx(c) {
    return { ...c, vars: { L: "L", n: "N", p: "N", prev: "N", last: "N", d: "N", x: "T", i: "int", start: "N", end: "N" }, muts: new Set() };
  }
  // {null}, {fn:name} and {e:<expression>} in step text become code in the chosen language
  function fmt(text, c) {
    if (!text) return "";
    return text.replace(/\{(null|fn:(\w+)(\([^)]*\))?|e:([^}]+)|s:([^}]+))\}/g, (m, a, fn, args, ex, st) => {
      if (a === "null") return "`" + NULLW[c.lang] + "`";
      if (fn) return "`" + (c.lang === "java" ? DL.camel(fn) : fn) + (args || "") + "`";
      if (ex) return "`" + expr(parseExpr(ex), textCtx(c)) + "`";
      return "`" + DL.stmt(DL.parseStmt(st), textCtx(c)).code.join(" ") + "`";
    });
  }
  const S = (say, ir, body) => (body ? { say, ir, body } : { say, ir });

  // ---------------------------------------------------------------- function specs
  const FNS = {
    is_segment: { params: [["start", "N"], ["end", "N"], ["n", "int"]], ret: "bool" },
    is_list: { params: [["L", "Lref"]], ret: "bool" },
    is_empty: { params: [["L", "Lref"]], ret: "bool" },
    insert_front: { params: [["L", "Lmut"], ["x", "T"]], ret: "void" },
    insert_back: { params: [["L", "Lmut"], ["x", "T"]], ret: "void" },
    delete_head: { params: [["L", "Lmut"]], ret: "T" },
    delete_node: { params: [["L", "Lmut"], ["x", "T"]], ret: "bool" },
    lookup: { params: [["L", "Lref"], ["x", "T"]], ret: "bool" },
    print_list: { params: [["L", "Lref"]], ret: "void" },
    show: { params: [["L", "Lref"]], ret: "void", harness: true, bounds: ["std::fmt::Display"] },
  };

  // ---------------------------------------------------------------- the functions, per layout
  const REQ = S("**Requires** (precondition): the list is valid when the function starts.", "assert is_list(L)");
  const ENS = S("**Ensures** (postcondition): the list is still valid when it returns.", "assert is_list(L)");
  const MORE = S("One more element.", "L.size += 1");
  const LESS = S("One element fewer.", "L.size -= 1");
  const NEWN = S("Make a new node `n` holding `x`. Its pointers start as {null}.", "let n = new(x)");

  function isSegment(v) {
    const loop = [
      S("Hitting {null} or `end` before `n` steps means the segment is shorter than `n`.", "if p == null or p == end", [S("", "return false")]),
    ];
    if (v.doubly) loop.push(S("Doubly linked: the next node must point back to `p`.", "if p.next != null and p.next.prev != p", [S("", "return false")]));
    loop.push(S("Step to the next node.", "p = p.next"));
    return [
      S("Start at `start`.", "let p = start"),
      S("Take exactly `n` steps. (Loop invariant: after k rounds, `p` is k nodes past `start`, and none of them was `end`.)", "repeat n", loop),
      S("After `n` steps we must stand exactly on `end`: then [`start`, `end`) holds exactly `n` nodes.", "return p == end"),
    ];
  }

  function isList(v) {
    const ifFalse = (say, cond) => S(say, `if ${cond}`, [S("", "return false")]);
    const k = `${v.dummy ? "D" : "N"}${v.doubly ? "D" : "S"}${v.circular ? "C" : "L"}`;
    switch (k) {
      case "NSL": return [
        S("Empty list: both ends must be {null}.", "if L.size == 0", [S("", "return L.head == null and L.tail == null")]),
        ifFalse("Not empty: both ends must exist.", "L.head == null or L.tail == null"),
        ifFalse("Nothing comes after the last node.", "L.tail.next != null"),
        S("The segment [`L.head`, `L.tail`) holds the other `L.size - 1` nodes.", "return is_segment(L.head, L.tail, L.size - 1)"),
      ];
      case "NDL": return [
        S("Empty list: both ends must be {null}.", "if L.size == 0", [S("", "return L.head == null and L.tail == null")]),
        ifFalse("Not empty: both ends must exist.", "L.head == null or L.tail == null"),
        ifFalse("Nothing comes before the first node or after the last.", "L.head.prev != null or L.tail.next != null"),
        S("[`L.head`, `L.tail`) holds the other `L.size - 1` nodes, every one linked both ways.", "return is_segment(L.head, L.tail, L.size - 1)"),
      ];
      case "NSC": return [
        S("Empty list: no last node.", "if L.size == 0", [S("", "return L.tail == null")]),
        ifFalse("Not empty: the last node must exist.", "L.tail == null"),
        S("Going around from the first node (`L.tail.next`), `L.size - 1` steps must land on `L.tail`.", "return is_segment(L.tail.next, L.tail, L.size - 1)"),
      ];
      case "NDC": return [
        S("Empty list: no first node.", "if L.size == 0", [S("", "return L.head == null")]),
        ifFalse("Not empty: the first node and its `prev` (the last node) must exist.", "L.head == null or L.head.prev == null"),
        ifFalse("The circle closes: the last node points back to the first.", "L.head.prev.next != L.head"),
        S("From `L.head`, `L.size - 1` steps must land on the last node, linked both ways.", "return is_segment(L.head, L.head.prev, L.size - 1)"),
      ];
      case "DSL": return [
        ifFalse("The dummy always exists and ends the list.", "L.dummy == null or L.dummy.next != null"),
        S("The segment [`L.head`, `L.dummy`) holds exactly `L.size` nodes.", "return is_segment(L.head, L.dummy, L.size)"),
      ];
      case "DDL": return [
        ifFalse("The dummy always exists and ends the list.", "L.dummy == null or L.dummy.next != null"),
        ifFalse("Nothing comes before the first node (the dummy, when empty).", "L.head == null or L.head.prev != null"),
        S("The segment [`L.head`, `L.dummy`) holds exactly `L.size` nodes, linked both ways.", "return is_segment(L.head, L.dummy, L.size)"),
      ];
      case "DSC": return [
        ifFalse("The dummy always exists and always has a next (itself, when empty).", "L.dummy == null or L.dummy.next == null"),
        S("Around the circle, [`L.dummy.next`, `L.dummy`) holds exactly `L.size` nodes.", "return is_segment(L.dummy.next, L.dummy, L.size)"),
      ];
      case "DDC": return [
        ifFalse("The dummy always exists and always has a next (itself, when empty).", "L.dummy == null or L.dummy.next == null"),
        ifFalse("The first node points back to the dummy.", "L.dummy.next.prev != L.dummy"),
        S("Around the circle, [`L.dummy.next`, `L.dummy`) holds exactly `L.size` nodes, linked both ways.", "return is_segment(L.dummy.next, L.dummy, L.size)"),
      ];
    }
    throw new Error(k);
  }

  function insertFront(v) {
    const k = `${v.dummy ? "D" : "N"}${v.doubly ? "D" : "S"}${v.circular ? "C" : "L"}`;
    const body = {
      NSL: [
        S("Link `n` in front of the current first node ({null} if the list is empty).", "n.next = L.head"),
        S("`n` is the new first node.", "L.head = n"),
        S("If the list was empty, `n` is also the last node.", "if L.tail == null", [S("", "L.tail = n")]),
      ],
      NDL: [
        S("Link `n` in front of the current first node.", "n.next = L.head"),
        S("If there is a first node, it now points back to `n`…", "if L.head != null", [S("", "L.head.prev = n")]),
        S("…otherwise the list was empty, and `n` is also the last node.", "else", [S("", "L.tail = n")]),
        S("`n` is the new first node.", "L.head = n"),
      ],
      NSC: [
        S("Empty list: `n` is a circle of one (it points to itself), and it is the last node.", "if L.tail == null", [S("", "n.next = n"), S("", "L.tail = n")]),
        S("Otherwise splice `n` in right after the last node. Since {e:L.tail.next} is the first node, \"after the last\" is the front.", "else",
          [S("`n` points to the old first node.", "n.next = L.tail.next"), S("The last node points to `n`.", "L.tail.next = n")]),
      ],
      NDC: [
        S("Empty list: `n` is a circle of one; both its pointers point to itself.", "if L.head == null", [S("", "n.next = n"), S("", "n.prev = n")]),
        S("Otherwise put `n` between the last node ({e:L.head.prev}) and the first node.", "else", [
          S("Name the last node.", "let last = L.head.prev"),
          S("`n` sits after `last`…", "n.prev = last"),
          S("…and before the old first node.", "n.next = L.head"),
          S("`last` points forward to `n`.", "last.next = n"),
          S("The old first node points back to `n`.", "L.head.prev = n"),
        ]),
        S("`n` is the new first node.", "L.head = n"),
      ],
      DSL: [
        S("Link `n` in front of the first node. No special case: even an empty list has a node to point at (the dummy).", "n.next = L.head"),
        S("`n` is the new first node.", "L.head = n"),
      ],
      DDL: [
        S("Link `n` in front of the first node.", "n.next = L.head"),
        S("The old first node (maybe the dummy) points back to `n`. It always exists, so no {null} check.", "L.head.prev = n"),
        S("`n` is the new first node.", "L.head = n"),
      ],
      DSC: [
        S("`n` points to the old first node (the dummy itself, if the list is empty).", "n.next = L.dummy.next"),
        S("The dummy points to `n`: the new first node.", "L.dummy.next = n"),
      ],
      DDC: [
        S("`n` goes between the dummy and the old first node: forward first…", "n.next = L.dummy.next"),
        S("…and back to the dummy.", "n.prev = L.dummy"),
        S("The old first node (maybe the dummy itself) points back to `n`.", "L.dummy.next.prev = n"),
        S("The dummy points forward to `n`.", "L.dummy.next = n"),
      ],
    }[k];
    return [REQ, NEWN, ...body, MORE, ENS];
  }

  function insertBack(v) {
    const k = `${v.dummy ? "D" : "N"}${v.doubly ? "D" : "S"}${v.circular ? "C" : "L"}`;
    const fill = [
      S("Write `x` into the dummy: the old dummy becomes the last element.", "L.dummy.data = x"),
      S("Make a fresh dummy `d`.", "let d = new(default)"),
    ];
    const body = {
      NSL: [NEWN,
        S("Empty list: `n` is also the first node…", "if L.tail == null", [S("", "L.head = n")]),
        S("…otherwise the old last node points to `n`.", "else", [S("", "L.tail.next = n")]),
        S("`n` is the new last node.", "L.tail = n")],
      NDL: [NEWN,
        S("`n` points back to the old last node ({null} if empty).", "n.prev = L.tail"),
        S("If there is a last node, it points forward to `n`…", "if L.tail != null", [S("", "L.tail.next = n")]),
        S("…otherwise `n` is also the first node.", "else", [S("", "L.head = n")]),
        S("`n` is the new last node.", "L.tail = n")],
      NSC: [NEWN,
        S("The same splice as {fn:insert_front}: empty → a circle of one…", "if L.tail == null", [S("", "n.next = n")]),
        S("…otherwise `n` goes right after the last node.", "else", [S("", "n.next = L.tail.next"), S("", "L.tail.next = n")]),
        S("The one difference from {fn:insert_front}: `n` becomes the last node.", "L.tail = n")],
      NDC: [NEWN,
        S("Empty list: a circle of one, and the first node.", "if L.head == null", [S("", "n.next = n"), S("", "n.prev = n"), S("", "L.head = n")]),
        S("Otherwise the same splice as {fn:insert_front}, between the last node and `L.head`; this time `L.head` stays.", "else", [
          S("", "let last = L.head.prev"), S("", "n.prev = last"), S("", "n.next = L.head"), S("", "last.next = n"), S("", "L.head.prev = n")])],
      DSL: [...fill,
        S("The new last element points to the new dummy.", "L.dummy.next = d"),
        S("`d` is the dummy now. No {null} check and no tail pointer needed.", "L.dummy = d")],
      DDL: [...fill,
        S("`d` points back to the new last element.", "d.prev = L.dummy"),
        S("The new last element points to `d`.", "L.dummy.next = d"),
        S("`d` is the dummy now.", "L.dummy = d")],
      DSC: [...fill,
        S("`d` continues the circle to the first node (the old dummy itself, if the list was empty).", "d.next = L.dummy.next"),
        S("The new last element points to `d`.", "L.dummy.next = d"),
        S("`d` is the dummy now.", "L.dummy = d")],
      DDC: [NEWN,
        S("`n` goes between the last node ({e:L.dummy.prev}) and the dummy: back first…", "n.prev = L.dummy.prev"),
        S("…and forward to the dummy.", "n.next = L.dummy"),
        S("The old last node (maybe the dummy itself) points forward to `n`.", "L.dummy.prev.next = n"),
        S("The dummy points back to `n`: the new last node.", "L.dummy.prev = n")],
    }[k];
    return [REQ, ...body, MORE, ENS];
  }

  function deleteHead(v) {
    const k = `${v.dummy ? "D" : "N"}${v.doubly ? "D" : "S"}${v.circular ? "C" : "L"}`;
    const take = S("Save the element and free the node.", "let x = take n");
    const body = {
      NSL: [S("`n` is the node to remove.", "let n = L.head"),
        S("The second node ({null} if none) is the new first.", "L.head = n.next"),
        S("If the list became empty, there's no last node either.", "if L.head == null", [S("", "L.tail = null")])],
      NDL: [S("`n` is the node to remove.", "let n = L.head"),
        S("The second node is the new first.", "L.head = n.next"),
        S("If there is one, nothing comes before it now…", "if L.head != null", [S("", "L.head.prev = null")]),
        S("…otherwise the list is empty.", "else", [S("", "L.tail = null")])],
      NSC: [S("`n` is the first node: the one after the last.", "let n = L.tail.next"),
        S("If it was the only node, the list is empty…", "if n == L.tail", [S("", "L.tail = null")]),
        S("…otherwise the last node now points past `n`.", "else", [S("", "L.tail.next = n.next")])],
      NDC: [S("`n` is the node to remove.", "let n = L.head"),
        S("If it was the only node, the list is empty…", "if n.next == n", [S("", "L.head = null")]),
        S("…otherwise join its neighbours and move `L.head` on.", "else", [
          S("The last node points forward past `n`.", "n.prev.next = n.next"),
          S("The second node points back to the last.", "n.next.prev = n.prev"),
          S("The second node is the new first.", "L.head = n.next")])],
      DSL: [S("`n` is the node to remove.", "let n = L.head"),
        S("The next node is the new first. No special case: if `n` was the last element, `L.head` becomes the dummy.", "L.head = n.next")],
      DDL: [S("`n` is the node to remove.", "let n = L.head"),
        S("The next node (maybe the dummy) is the new first.", "L.head = n.next"),
        S("Nothing comes before it now.", "L.head.prev = null")],
      DSC: [S("`n` is the first node.", "let n = L.dummy.next"),
        S("The dummy points past it.", "L.dummy.next = n.next")],
      DDC: [S("`n` is the first node.", "let n = L.dummy.next"),
        S("The dummy points forward past `n`…", "L.dummy.next = n.next"),
        S("…and the next node (maybe the dummy) points back to the dummy.", "n.next.prev = L.dummy")],
    }[k];
    return [REQ, S("**Requires**: the list is not empty (there is a head to delete).", "assert not is_empty(L)"),
      ...body, LESS, take, ENS, S("Return the element.", "return x")];
  }

  function deleteNode(v) {
    const k = `${v.dummy ? "D" : "N"}${v.doubly ? "D" : "S"}${v.circular ? "C" : "L"}`;
    const notFound = (cond, say) => S(say || "Not found: nothing changed, so there is nothing to re-check. Return false.", `if ${cond}`, [S("", "return false")]);
    const tail = [LESS, S("Free the removed node.", "free p"), ENS, S("Found and removed.", "return true")];
    const body = {
      NSL: [
        S("`prev` trails one node behind `p`; there is no node before the first.", "let prev = null"),
        S("Start at the first node.", "let p = L.head"),
        S("Walk until `p` holds `x` or falls off the end.", "while p != null and p.data != x", [S("", "prev = p"), S("", "p = p.next")]),
        notFound("p == null"),
        S("Unlink `p`: if it is the first node, `L.head` skips it…", "if prev == null", [S("", "L.head = p.next")]),
        S("…otherwise the node before it skips it.", "else", [S("", "prev.next = p.next")]),
        S("If `p` was the last node, the node before it is the new last.", "if p == L.tail", [S("", "L.tail = prev")]),
      ],
      NDL: [
        S("Start at the first node.", "let p = L.head"),
        S("Walk until `p` holds `x` or falls off the end. No `prev` variable: {e:p.prev} is right there.", "while p != null and p.data != x", [S("", "p = p.next")]),
        notFound("p == null"),
        S("Unlink `p` going forward: the node before skips it (or `L.head` does, if `p` was first)…", "if p.prev != null", [S("", "p.prev.next = p.next")]),
        S("", "else", [S("", "L.head = p.next")]),
        S("…and going back: the node after points back past it (or `L.tail` moves, if `p` was last). This unlinking is O(1).", "if p.next != null", [S("", "p.next.prev = p.prev")]),
        S("", "else", [S("", "L.tail = p.prev")]),
      ],
      NSC: [
        notFound("L.tail == null", "Empty list: not found."),
        S("`prev` starts at the last node, the one before the first.", "let prev = L.tail"),
        S("`p` starts at the first node.", "let p = L.tail.next"),
        S("Count the steps: a circle has no {null} to stop at.", "let i = 0"),
        S("Walk at most once around, until `p` holds `x`.", "while i < L.size and p.data != x", [S("", "prev = p"), S("", "p = p.next"), S("", "i += 1")]),
        notFound("i == L.size", "Went all the way around: not found. Return false."),
        S("If `p` was the only node, the list is empty…", "if p == prev", [S("", "L.tail = null")]),
        S("…otherwise the node before skips `p`, and if `p` was the last node, that node is the new last.", "else", [
          S("", "prev.next = p.next"), S("", "if p == L.tail", [S("", "L.tail = prev")])]),
      ],
      NDC: [
        S("Start at the first node ({null} when the list is empty: then the walk below never starts).", "let p = L.head"),
        S("Count the steps: a circle has no {null} to stop at.", "let i = 0"),
        S("Walk at most once around, until `p` holds `x`.", "while i < L.size and p.data != x", [S("", "p = p.next"), S("", "i += 1")]),
        notFound("i == L.size", "Went all the way around: not found. Return false."),
        S("If `p` was the only node, the list is empty…", "if p.next == p", [S("", "L.head = null")]),
        S("…otherwise join its neighbours (O(1)), and move `L.head` on if `p` was first.", "else", [
          S("", "p.prev.next = p.next"), S("", "p.next.prev = p.prev"), S("", "if p == L.head", [S("", "L.head = p.next")])]),
      ],
      DSL: [
        S("`prev` trails one node behind `p`; there is no node before the first.", "let prev = null"),
        S("Start at the first node.", "let p = L.head"),
        S("Walk until `p` holds `x` or reaches the dummy (the end).", "while p != L.dummy and p.data != x", [S("", "prev = p"), S("", "p = p.next")]),
        notFound("p == L.dummy"),
        S("Unlink `p`: `L.head` or the node before skips it. `p` is never the dummy, so {e:p.next} always exists.", "if prev == null", [S("", "L.head = p.next")]),
        S("", "else", [S("", "prev.next = p.next")]),
      ],
      DDL: [
        S("Start at the first node.", "let p = L.head"),
        S("Walk until `p` holds `x` or reaches the dummy.", "while p != L.dummy and p.data != x", [S("", "p = p.next")]),
        notFound("p == L.dummy"),
        S("The node after `p` always exists (at worst, the dummy): it points back past `p`.", "p.next.prev = p.prev"),
        S("Going forward, the node before skips `p` (or `L.head` does, if `p` was first).", "if p.prev != null", [S("", "p.prev.next = p.next")]),
        S("", "else", [S("", "L.head = p.next")]),
      ],
      DSC: [
        S("`prev` starts at the dummy: every node, even the first, has a node before it.", "let prev = L.dummy"),
        S("Start at the first node.", "let p = L.dummy.next"),
        S("Walk until `p` holds `x` or comes back around to the dummy.", "while p != L.dummy and p.data != x", [S("", "prev = p"), S("", "p = p.next")]),
        notFound("p == L.dummy"),
        S("The node before skips `p`. No special case for the first or last node.", "prev.next = p.next"),
      ],
      DDC: [
        S("Start at the first node.", "let p = L.dummy.next"),
        S("Walk until `p` holds `x` or comes back around to the dummy.", "while p != L.dummy and p.data != x", [S("", "p = p.next")]),
        notFound("p == L.dummy"),
        S("Join its neighbours, which always exist (maybe the dummy): forward…", "p.prev.next = p.next"),
        S("…and back. No special cases at all.", "p.next.prev = p.prev"),
      ],
    }[k];
    return [REQ, ...body, ...tail];
  }

  function lookup(v) {
    const found = S("Found it.", "if p.data == x", [S("", "return true")]);
    const step = S("Next node.", "p = p.next");
    if (!v.circular || v.dummy) {
      const first = v.dummy ? (v.circular ? "L.dummy.next" : "L.head") : "L.head";
      const cond = v.dummy ? "p != L.dummy" : "p != null";
      return [REQ, S("Start at the first node.", `let p = ${first}`),
        S(v.dummy ? "Visit every node up to, not including, the dummy." : "Visit every node until {null}.", `while ${cond}`, [found, step]),
        S("Checked them all: not there.", "return false")];
    }
    const first = v.doubly ? "L.head" : "L.tail.next";
    return [REQ,
      ...(v.doubly ? [] : [S("Empty list: not there.", "if L.tail == null", [S("", "return false")])]),
      S("Start at the first node.", `let p = ${first}`),
      S("Visit exactly `L.size` nodes: a circle has no {null} to stop at.", "repeat L.size", [found, step]),
      S("Checked them all: not there.", "return false")];
  }

  function printList(v) {
    const first = v.dummy ? (v.circular ? "L.dummy.next" : "L.head") : v.circular ? (v.doubly ? "L.head" : "L.tail.next") : "L.head";
    const visit = [
      S("Before every element except the first, print a comma and a space.", `if p != ${first}`, [S("", 'print ", "')]),
      S("Print the element.", "print p.data"),
      S("Next node.", "p = p.next"),
    ];
    const open = S("Print `[` (no newline).", 'print "["');
    const close = S("Print `]` and end the line.", 'println "]"');
    if (!v.circular || v.dummy) {
      const cond = v.dummy ? "p != L.dummy" : "p != null";
      return [open, S("Start at the first node.", `let p = ${first}`),
        S(v.dummy ? "Visit every node up to the dummy." : "Visit every node until {null}.", `while ${cond}`, visit), close];
    }
    if (v.doubly) {
      return [open, S("Start at the first node.", "let p = L.head"),
        S("Visit exactly `L.size` nodes (none when empty, so `p` may be {null}).", "repeat L.size", visit), close];
    }
    return [open,
      S("Only a non-empty list has nodes to print.", "if L.tail != null", [
        S("Start at the first node.", "let p = L.tail.next"),
        S("Visit exactly `L.size` nodes.", "repeat L.size",
          [S("Comma before every element but the first.", "if p != L.tail.next", [S("", 'print ", "')]), S("", "print p.data"), S("", "p = p.next")]),
      ]), close];
  }

  // ---------------------------------------------------------------- type declarations (step 1 and 2), per language
  function nodeSteps(c, v) {
    const g = c.generic, L = c.lang;
    const T = c.T;
    if (L === "py") {
      const hint = g ? "Node[T] | None" : "Node | None";
      return [{ say: g ? "Declare the node class, generic over the element type: `class Node[T]`." : "Declare the node class `Node`.",
        code: [g ? "class Node[T]:" : "class Node:"], body: [
          { say: "Its constructor `__init__` takes the element.", code: [`def __init__(self, data: ${T}) -> None:`], body: [
            { say: "Store the element.", code: ["self.data = data"] },
            { say: "`next`: the following node, `None` (null) until something links it.", code: [`self.next: ${hint} = None`] },
            ...(v.doubly ? [{ say: "`prev`: the node before, also `None` for now.", code: [`self.prev: ${hint} = None`] }] : []),
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
            ...(v.doubly ? [{ say: "A pointer to the previous node, null by default.", code: [`${P} prev = nullptr;`] }] : []),
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
          ...(v.doubly ? [{ say: "A reference to the previous node (also `null` at first).", code: [`${P} prev;`] }] : []),
          { say: "A constructor that stores the element.", code: [`Node(${T} data) {`], close: ["}"], body: [{ say: "", code: ["this.data = data;"] }] },
        ] }];
    }
    const N = g ? "Node<T>" : "Node";
    const init = v.doubly ? "Node { data, next: ptr::null_mut(), prev: ptr::null_mut() }" : "Node { data, next: ptr::null_mut() }";
    return [
      { say: "At the very top: allow capital variable names (the list is called `L`), and import `std::ptr` for null pointers.",
        code: ["#![allow(non_snake_case)]", "use std::ptr;"] },
      { say: g ? "Declare the struct `Node<T>`, generic over the element type." : "Declare the struct `Node`.",
        code: [`struct ${N} {`], close: ["}"], body: [
          { say: "A field for the element.", code: [`data: ${T},`] },
          { say: "A raw pointer to the next node (`*mut`). `Box` allows one owner per node, but in this list a node can be pointed at from two places (a tail pointer, a `prev`, a dummy, a circle). That needs raw pointers; `std::collections::LinkedList` is built the same way.",
            code: [`next: *mut ${N},`] },
          ...(v.doubly ? [{ say: "A raw pointer to the previous node.", code: [`prev: *mut ${N},`] }] : []),
        ] },
      { say: "A constructor `Node::new` that puts the node on the heap and returns a raw pointer to it.",
        code: [g ? "impl<T> Node<T> {" : "impl Node {"], close: ["}"], body: [
          { say: "", code: [`fn new(data: ${T}) -> *mut ${N} {`], close: ["}"], body: [
            { say: "Box the node (a heap allocation) with null pointers, then turn the box into a raw pointer.", code: [`Box::into_raw(Box::new(${init}))`] }] },
        ] },
    ];
  }

  function fieldSay(f, v) {
    if (f === "size") return "`size`: the number of elements, so the length is O(1).";
    if (f === "head") return v.dummy ? "`head`: the first node, or the dummy itself when the list is empty."
      : v.circular ? "`head`: the first node ({null} when empty). Its `prev` is the last node, so one pointer reaches both ends." : "`head`: the first node ({null} when the list is empty).";
    if (f === "tail") return v.circular ? "`tail`: the last node ({null} when empty). The circle closes through it: its `next` is the first node, so one pointer reaches both ends."
      : "`tail`: the last node ({null} when empty). It makes inserting at the end O(1).";
    return v.circular ? "`dummy`: the dummy node. Its `next` is the first node, and the last node points back to it. Empty = the dummy pointing to itself."
      : "`dummy`: the dummy node at the end. It holds no element; the list is the segment [`L.head`, `L.dummy`).";
  }

  function listSteps(c, v) {
    const g = c.generic, L = c.lang;
    const fs = fields(v);
    const dflt = !g ? "0" : { py: "None", cpp: "T{}", java: "null", rs: "T::default()" }[L];
    const ctorSay = v.dummy ? `The constructor makes an empty list: one dummy node${v.circular ? " pointing to itself" : ""}${!v.circular ? ", which is also the head" : ""}.` : "";
    if (L === "py") {
      const hint = g ? "Node[T] | None" : "Node | None";
      const lines = [];
      if (v.dummy) {
        lines.push({ say: "Make the dummy. Its element is a placeholder that is never read.", code: [g ? "self.dummy: Node[T] = Node(None)" : "self.dummy = Node(0)"] });
        if (!v.circular) lines.push({ say: fieldSay("head", v), code: ["self.head = self.dummy"] });
        else {
          lines.push({ say: "Empty circle: the dummy points to itself.", code: ["self.dummy.next = self.dummy"] });
          if (v.doubly) lines.push({ say: "", code: ["self.dummy.prev = self.dummy"] });
        }
      } else fs.forEach((f) => lines.push({ say: fieldSay(f, v), code: [`self.${f}: ${hint} = None`] }));
      lines.push({ say: fieldSay("size", v), code: ["self.size = 0"] });
      return [{ say: g ? "Declare the list class `LinkedList[T]`." : "Declare the list class `LinkedList`.", code: [g ? "class LinkedList[T]:" : "class LinkedList:"], body: [
        { say: v.dummy ? ctorSay : "The constructor makes an empty list.", code: ["def __init__(self) -> None:"], body: lines }] }];
    }
    if (L === "cpp") {
      const P = g ? "Node<T>*" : "Node*";
      const body = [];
      if (!v.dummy) fs.forEach((f) => body.push({ say: fieldSay(f, v), code: [`${P} ${f} = nullptr;`] }));
      else fs.forEach((f) => body.push({ say: fieldSay(f, v), code: [`${P} ${f};`] }));
      body.push({ say: fieldSay("size", v), code: ["int size = 0;"] });
      if (v.dummy) {
        const ctor = [{ say: "", code: [`dummy = new ${g ? "Node<T>" : "Node"}(${dflt});`] }];
        if (!v.circular) ctor.push({ say: "", code: ["head = dummy;"] });
        else { ctor.push({ say: "", code: ["dummy->next = dummy;"] }); if (v.doubly) ctor.push({ say: "", code: ["dummy->prev = dummy;"] }); }
        body.push({ say: ctorSay, code: ["LinkedList() {"], close: ["}"], body: ctor });
      }
      return [{ say: (g ? "Declare the list struct, a template over `T`." : "Declare the list struct `LinkedList`.") +
        (v.dummy ? "" : " No constructor needed: the default values already make an empty list."),
      code: g ? ["template <typename T>", "struct LinkedList {"] : ["struct LinkedList {"], close: ["};"], body }];
    }
    if (L === "java") {
      const P = g ? "Node<T>" : "Node";
      const body = fs.map((f) => ({ say: fieldSay(f, v), code: [`${P} ${f};`] }));
      body.push({ say: fieldSay("size", v) + " (Java starts it at 0.)", code: ["int size;"] });
      if (v.dummy) {
        const ctor = [{ say: "", code: [`dummy = new ${g ? "Node<>" : "Node"}(${dflt});`] }];
        if (!v.circular) ctor.push({ say: "", code: ["head = dummy;"] });
        else { ctor.push({ say: "", code: ["dummy.next = dummy;"] }); if (v.doubly) ctor.push({ say: "", code: ["dummy.prev = dummy;"] }); }
        body.push({ say: ctorSay, code: ["LinkedList() {"], close: ["}"], body: ctor });
      }
      return [{ say: (g ? "Declare the class `LinkedList<T>`." : "Declare the class `LinkedList`.") +
        (v.dummy ? "" : " Fields start as `null` and 0, which is already an empty list.") +
        " Every function from here on goes inside this class, as a `static` method that takes the list `L`.",
      code: [g ? "class LinkedList<T> {" : "class LinkedList {"], close: ["}"], body, deferClose: true }];   // its brace ends the file
    }
    const N = g ? "Node<T>" : "Node";
    const body = fs.map((f) => ({ say: fieldSay(f, v), code: [`${f}: *mut ${N},`] }));
    body.push({ say: fieldSay("size", v), code: ["size: usize,"] });
    let ctor;
    if (!v.dummy) ctor = [{ say: "Both ends null, size 0.", code: [`LinkedList { ${fs.map((f) => `${f}: ptr::null_mut()`).join(", ")}, size: 0 }`] }];
    else {
      ctor = [{ say: "Make the dummy (its element is a placeholder that is never read).", code: [`let d = Node::new(${dflt});`] }];
      if (v.circular) ctor.push({ say: "Empty circle: the dummy points to itself.", code: [v.doubly ? "unsafe { (*d).next = d; (*d).prev = d; }" : "unsafe { (*d).next = d; }"] });
      ctor.push({ say: "", code: [`LinkedList { ${fs.map((f) => `${f}: d`).join(", ")}, size: 0 }`] });
    }
    const implHead = !g ? "impl LinkedList {" : v.dummy ? "impl<T: Default> LinkedList<T> {" : "impl<T> LinkedList<T> {";
    return [
      { say: g ? "Declare the struct `LinkedList<T>`." : "Declare the struct `LinkedList`.", code: [g ? "struct LinkedList<T> {" : "struct LinkedList {"], close: ["}"], body },
      { say: (v.dummy ? ctorSay : "A constructor `LinkedList::new()` that makes an empty list.") + (v.dummy && g ? " The dummy needs some value of type `T`, hence `T: Default`." : ""),
        code: [implHead], close: ["}"], body: [{ say: "", code: ["fn new() -> Self {"], close: ["}"], body: ctor }] },
    ];
  }

  // ---------------------------------------------------------------- steps
  function contractNote(lang) {
    return {
      py: "Python runs `assert` unless started with `python -O`.",
      cpp: "C++ runs `assert` unless compiled with `-DNDEBUG`.",
      java: "Java skips `assert` unless started with `java -ea`; the checker turns assertions on.",
      rs: "`debug_assert!` runs in debug builds (what the checker uses) and disappears from release builds, like 15-122's contracts outside `-d` mode.",
    }[lang];
  }

  function steps(o) {
    const v = variant(o);
    const fn = (name, stepsFn, headerSay) => ({ name, steps: stepsFn(v), headerSay });
    const first = { NSL: "`L.head`", NDL: "`L.head`", NSC: "{e:L.tail.next}", NDC: "`L.head`", DSL: "`L.head`", DDL: "`L.head`", DSC: "{e:L.dummy.next}", DDC: "{e:L.dummy.next}" };
    const k = `${v.dummy ? "D" : "N"}${v.doubly ? "D" : "S"}${v.circular ? "C" : "L"}`;
    return [
      { id: "node", title: "The node",
        goal: "A linked list is a chain of nodes. Each node holds one element (`data`) and a pointer to the next node" +
          (v.doubly ? ", plus one to the previous node" : "") + ". A pointer that points nowhere is null, and every pointer starts out null.",
        decl: (c) => nodeSteps(c, v), tests: ["node"] },
      { id: "list", title: "The list and its contracts",
        goal: "Before writing any operation, write down what a correct list looks like, as code that checks it. These are the **contracts**:\n\n" +
          "* {fn:is_segment(start, end, n)}: following `next` from `start` reaches `end` after exactly `n` nodes. The nodes [`start`, `end`) form a **segment**: `start` included, `end` excluded.\n" +
          "* {fn:is_list(L)}: the **representation invariant**, true exactly when `L` is a well-formed list.\n" +
          "* Every operation then starts with {s:assert is_list(L)} (what it **requires**) and checks it again before returning (what it **ensures**). If a function breaks the list, the very next check stops the program right there.\n\n" +
          `The list is \`L\`. ${diagram(o).caption}`,
        decl: (c) => listSteps(c, v),
        fns: [fn("is_segment", isSegment, "Write {fn:is_segment}: does the segment [`start`, `end`) hold exactly `n` nodes?"),
          fn("is_list", isList, "Write {fn:is_list}, the representation invariant."),
          { name: "is_empty", steps: [S("The list is empty exactly when its size is 0.", "return L.size == 0")], headerSay: "Write {fn:is_empty}." }],
        contractNote: true, tests: ["node", "list"] },
      { id: "insert_front", title: "Insert at the start",
        goal: `{fn:insert_front(L, x)} puts \`x\` before the first element (${first[k]}). O(1): a fixed number of pointer changes, no loop.`,
        fns: [fn("insert_front", insertFront, "Write the header: {fn:insert_front} takes the list and the new element.")], tests: ["node", "list", "insert_front"] },
      { id: "insert_back", title: "Insert at the end",
        goal: "{fn:insert_back(L, x)} puts `x` after the last element, also in O(1). " + ({
          NSL: "`L.tail` gives the last node directly.", NDL: "`L.tail` gives the last node directly.",
          NSC: "`L.tail` gives the last node directly.", NDC: "{e:L.head.prev} gives the last node directly.",
          DSL: "The trick with a dummy at the end: write `x` into the dummy, then add a fresh dummy after it.",
          DDL: "The trick with a dummy at the end: write `x` into the dummy, then add a fresh dummy after it.",
          DSC: "The trick with a dummy: write `x` into the dummy, then add a fresh dummy after it.",
          DDC: "{e:L.dummy.prev} is the last node, so this is the mirror image of {fn:insert_front}.",
        }[k]),
        fns: [fn("insert_back", insertBack, "Write the header: {fn:insert_back} takes the list and the new element.")], tests: ["node", "list", "insert_front", "insert_back"] },
      { id: "delete_head", title: "Delete the head",
        goal: "{fn:delete_head(L)} removes the first element and returns it. It **requires** a non-empty list, so add that to the contract. O(1).",
        fns: [fn("delete_head", deleteHead, "Write the header: {fn:delete_head} returns the removed element.")], tests: ["node", "list", "insert_front", "insert_back", "delete_head"] },
      { id: "delete_node", title: "Delete a node",
        goal: "{fn:delete_node(L, x)} removes the first node holding `x` and returns true, or returns false if `x` isn't there. Finding the node is O(n); " +
          (v.doubly ? "unlinking it is O(1), because a doubly linked node knows both neighbours." : "unlinking it needs the node *before* it, so the walk keeps a `prev` pointer one step behind."),
        fns: [fn("delete_node", deleteNode, "Write the header: {fn:delete_node} takes the list and the element to remove, and returns a bool.")],
        tests: ["node", "list", "insert_front", "insert_back", "delete_head", "delete_node"] },
      { id: "lookup", title: "Lookup",
        goal: "{fn:lookup(L, x)} returns true if some node holds `x`. Visit the nodes one by one: O(n). It doesn't change the list, so only the precondition is needed.",
        fns: [fn("lookup", lookup, "Write the header: {fn:lookup} takes the list and the element, and returns a bool.")],
        tests: ["node", "list", "insert_front", "insert_back", "delete_head", "delete_node", "lookup"] },
      { id: "print_list", title: "Traverse and print",
        goal: "{fn:print_list(L)} prints the elements in order, like `[3, 5, 7]`, and `[]` when empty, then ends the line. The same walk as {fn:lookup}, visiting every node: O(n).",
        fns: [fn("print_list", printList, "Write the header: {fn:print_list} takes the list and prints it.")],
        tests: ["node", "list", "insert_front", "insert_back", "delete_head", "delete_node", "lookup", "print_list"] },
    ];
  }

  // ---------------------------------------------------------------- rendering a tutorial for one set of options
  function context(o, overrides = {}) {
    const T = o.type === "int" ? { py: "int", cpp: "int", java: "int", rs: "i32" }[o.lang] : "T";
    return { lang: o.lang, generic: o.type === "generic", T, fns: FNS, vars: {}, muts: new Set(), ...overrides };
  }

  // rendered steps: [{id, title, goal, items: [display micro-steps], code (the step's code), names}]
  function render(o, list = steps(o)) {
    const c = context(o);
    return list.map((st) => {
      const items = [];
      const parts = [];
      if (st.decl) {
        const d = DL.toDisplay(st.decl(c), c);
        d.forEach((m) => { m.say = fmt(m.say, c); fmtDeep(m.body, c); });
        items.push(...d);
        parts.push(d.map((m) => DL.emit([m]).join("\n")).join("\n\n"));
      }
      for (const f of st.fns || []) {
        const r = DL.renderFunction(f, c);
        r.display.forEach((m) => { m.say = fmt(m.say, c); fmtDeep(m.body, c); });
        items.push({ say: fmt(f.headerSay, c), group: true, fn: f.name, body: r.display.slice(0), code: [], close: [] });
        parts.push(r.code);
      }
      return { id: st.id, title: st.title, goal: fmt(st.goal, c), items, code: parts.join("\n\n"), contractNote: st.contractNote ? contractNote(o.lang) : null,
        names: (st.fns || []).map((f) => fname(f.name, o.lang)) };
    });
  }
  function fmtDeep(list, c) { for (const m of list) { m.say = fmt(m.say, c); fmtDeep(m.body, c); } }
  const fname = (n, lang) => (lang === "java" ? DL.camel(n) : n);

  // the reference solution for steps [0, upto): the user's file, as the tutorial builds it
  function program(o, upto = 8, list) {
    const r = render(o, list).slice(0, upto);
    if (o.lang !== "java") return r.map((s) => s.code).join("\n\n") + "\n";
    // Java: the functions go inside class LinkedList, before its closing brace
    const parts = r.map((s) => s.code);
    if (upto < 2) return parts.join("\n\n") + "\n";
    const step2 = parts[1];
    const fnStart = step2.indexOf("\n\nstatic ");
    const classPart = step2.slice(0, fnStart);
    const fns = [step2.slice(fnStart + 2), ...parts.slice(2)].join("\n\n");
    const closeAt = classPart.lastIndexOf("}");
    return `${parts[0]}\n\n${classPart.slice(0, closeAt)}\n${DL.indent(fns, 1)}\n}\n`;
  }

  // ---------------------------------------------------------------- tests
  const VALUES = { int: [10, 20, 30, 40, 50, 60], generic: ["ant", "bee", "cat", "dog", "eel", "fox"] };
  const ABSENT = { int: 99, generic: "yak" };

  function breaks(v) {
    const k = `${v.dummy ? "D" : "N"}${v.doubly ? "D" : "S"}${v.circular ? "C" : "L"}`;
    const common = [["L.size += 1", "L.size is one too big"]];
    return common.concat({
      NSL: [["L.tail = L.head", "L.tail points at the first node"], ["L.tail.next = L.head", "the last node points back to the first"], ["L.head = null", "L.head is null"],
        ["L.tail = null", "L.tail is null"]],
      NDL: [["L.tail = L.head", "L.tail points at the first node"], ["L.head.next.prev = null", "the second node's prev is null"], ["L.head.prev = L.tail", "L.head.prev is not null"], ["L.tail.next = L.head", "the last node points back to the first"],
        ["L.head = null", "L.head is null"], ["L.tail = null", "L.tail is null"]],
      NSC: [["L.tail.next = L.tail", "the last node points to itself"], ["L.tail = null", "L.tail is null"], ["L.size -= 1", "L.size is one too small"]],
      NDC: [["L.head.prev = L.head", "L.head.prev points at the first node"], ["L.head.next.prev = null", "the second node's prev is null"], ["L.head.prev.next = L.head.next", "the last node's next skips the first node"],
        ["L.head.prev = null", "L.head.prev is null"], ["L.head = null", "L.head is null"]],
      DSL: [["L.dummy = L.head", "L.dummy points at the first node"], ["L.dummy.next = L.head", "the dummy's next is not null"], ["L.size -= 1", "L.size is one too small"],
        ["L.dummy = null", "L.dummy is null"]],
      DDL: [["L.dummy = L.head", "L.dummy points at the first node"], ["L.head.next.prev = null", "the second node's prev is null"], ["L.head.prev = L.dummy", "L.head.prev is not null"],
        ["L.dummy = null", "L.dummy is null"], ["L.dummy.next = L.head", "the dummy's next is not null"]],
      DSC: [["L.dummy.next = L.dummy", "the dummy's next skips every node"], ["L.size -= 1", "L.size is one too small"], ["L.dummy.next.next = L.dummy", "the first node's next skips the others"],
        ["L.dummy = null", "L.dummy is null"]],
      DDC: [["L.dummy.next.prev = null", "the first node's prev is null"], ["L.dummy.prev.next = L.dummy.next", "the last node's next skips the dummy"], ["L.size -= 1", "L.size is one too small"],
        ["L.dummy.next.next.prev = L.dummy", "the second node's prev points to the dummy"], ["L.dummy.next = null", "the dummy's next is null"],
        ["L.dummy = null", "L.dummy is null"]],
    }[k]);
  }

  // the segment that holds the elements: is_segment(first, end, n) is true for a valid list
  function segment(v) {
    const k = `${v.dummy ? "D" : "N"}${v.doubly ? "D" : "S"}${v.circular ? "C" : "L"}`;
    return { NSL: ["L.head", "L.tail", -1], NDL: ["L.head", "L.tail", -1], NSC: ["L.tail.next", "L.tail", -1], NDC: ["L.head", "L.head.prev", -1],
      DSL: ["L.head", "L.dummy", 0], DDL: ["L.head", "L.dummy", 0], DSC: ["L.dummy.next", "L.dummy", 0], DDC: ["L.dummy.next", "L.dummy", 0] }[k];
  }

  // the harness's own view of the list: the elements, the size, and the first broken rule (if any)
  function showFn(v) {
    const k = `${v.dummy ? "D" : "N"}${v.doubly ? "D" : "S"}${v.circular ? "C" : "L"}`;
    const st = [S("", 'print "["')];
    let cond;
    if (k === "NSC") { st.push(S("", "let p = null"), S("", "if L.tail != null", [S("", "p = L.tail.next")])); cond = "p != null and count < L.size + 2 and (count == 0 or p != L.tail.next)"; }
    else if (k === "NDC") { st.push(S("", "let p = L.head")); cond = "p != null and count < L.size + 2 and (count == 0 or p != L.head)"; }
    else if (k === "DSC" || k === "DDC") { st.push(S("", "let p = null"), S("", "if L.dummy != null", [S("", "p = L.dummy.next")])); cond = "p != null and p != L.dummy and count < L.size + 2"; }
    else if (v.dummy) { st.push(S("", "let p = L.head")); cond = "p != null and p != L.dummy and count < L.size + 2"; }
    else { st.push(S("", "let p = L.head")); cond = "p != null and count < L.size + 2"; }
    st.push(S("", "let count = 0"),
      S("", `while ${cond}`, [S("", "if count > 0", [S("", 'print ", "')]), S("", "print p.data"), S("", "p = p.next"), S("", "count += 1")]),
      S("", 'print "] size="'), S("", "print L.size"));
    const check = (c, msg) => S("", `if not (${c})`, [S("", `print " BROKEN: ${msg}"`), S("", "println"), S("", "return")]);
    const seg = "walking next from the first node does not reach the end in exactly the right number of steps (check size, next" + (v.doubly ? " and prev" : "") + " pointers)";
    const checks = {
      NSL: [S("", "if L.size == 0", [check("L.head == null", "L.head must be null in an empty list"), check("L.tail == null", "L.tail must be null in an empty list")]),
        S("", "else", [check("L.head != null", "L.head is null but L.size is not 0"), check("L.tail != null", "L.tail is null but L.size is not 0"),
          check("L.tail.next == null", "L.tail.next is not null, so L.tail is not the last node"), check("is_segment(L.head, L.tail, L.size - 1)", seg)])],
      NDL: [S("", "if L.size == 0", [check("L.head == null", "L.head must be null in an empty list"), check("L.tail == null", "L.tail must be null in an empty list")]),
        S("", "else", [check("L.head != null", "L.head is null but L.size is not 0"), check("L.tail != null", "L.tail is null but L.size is not 0"),
          check("L.head.prev == null", "L.head.prev must be null"), check("L.tail.next == null", "L.tail.next is not null, so L.tail is not the last node"),
          check("is_segment(L.head, L.tail, L.size - 1)", seg)])],
      NSC: [S("", "if L.size == 0", [check("L.tail == null", "L.tail must be null in an empty list")]),
        S("", "else", [check("L.tail != null", "L.tail is null but L.size is not 0"), check("is_segment(L.tail.next, L.tail, L.size - 1)", seg)])],
      NDC: [S("", "if L.size == 0", [check("L.head == null", "L.head must be null in an empty list")]),
        S("", "else", [check("L.head != null", "L.head is null but L.size is not 0"), check("L.head.prev != null", "L.head.prev must point to the last node"),
          check("L.head.prev.next == L.head", "the last node's next must point back to L.head"), check("is_segment(L.head, L.head.prev, L.size - 1)", seg)])],
      DSL: [check("L.dummy != null", "L.dummy is null"), check("L.dummy.next == null", "the dummy's next must be null"), check("is_segment(L.head, L.dummy, L.size)", seg)],
      DDL: [check("L.dummy != null", "L.dummy is null"), check("L.dummy.next == null", "the dummy's next must be null"),
        check("L.head != null", "L.head is null"), check("L.head.prev == null", "L.head.prev must be null"), check("is_segment(L.head, L.dummy, L.size)", seg)],
      DSC: [check("L.dummy != null", "L.dummy is null"), check("L.dummy.next != null", "the dummy's next is null"), check("is_segment(L.dummy.next, L.dummy, L.size)", seg)],
      DDC: [check("L.dummy != null", "L.dummy is null"), check("L.dummy.next != null", "the dummy's next is null"),
        check("L.dummy.next.prev == L.dummy", "the first node's prev must point to the dummy"), check("is_segment(L.dummy.next, L.dummy, L.size)", seg)],
    }[k];
    return [...st, ...checks, S("", "println")];
  }

  // Test groups per step. An op is [kind, value]; each group starts with a fresh list.
  function testGroups(names, o, rnd) {
    const v = variant(o);
    const [a, b, c, d, e, f] = VALUES[o.type];
    const z = ABSENT[o.type];
    const G = {
      node: [[["node", a]], [["node", f]]],
      list: [
        [["new"], ["is_empty"], ["is_list"]],
        [["new"], ["fill", [a, b, c]], ["is_empty"], ["is_list"], ["segment", 0], ["segment", -1], ["segment", 1], ["segment", 2], ...(v.circular ? [["segment", "ring"]] : [])],
        [["new"], ["fill", [d]], ["is_list"], ["segment", 0]],
        ...breaks(v).map((_, i) => [["new"], ["fill", [a, b, c]], ["break", i]]),
      ],
      insert_front: [[["new"], ["insert_front", a], ["insert_front", b], ["insert_front", c], ["is_list"]]],
      insert_back: [
        [["new"], ["insert_back", a], ["insert_back", b], ["insert_front", c], ["insert_back", d], ["is_list"]],
        [["new"], ["insert_front", e], ["insert_back", f]],
      ],
      delete_head: [
        [["new"], ["insert_back", a], ["insert_back", b], ["insert_back", c], ["delete_head"], ["delete_head"], ["delete_head"], ["is_empty"],
          ["insert_front", d], ["insert_back", e], ["delete_head"], ["insert_back", f], ["delete_head"], ["delete_head"], ["insert_back", a]],
      ],
      delete_node: [
        [["new"], ["delete_node", a], ["insert_back", a], ["insert_back", b], ["insert_back", c], ["insert_back", d], ["insert_back", b],
          ["delete_node", b], ["delete_node", z], ["delete_node", a], ["delete_node", b], ["delete_node", d], ["delete_node", c], ["delete_node", c],
          ["insert_back", e], ["insert_front", f], ["delete_node", e], ["insert_back", a]],
      ],
      lookup: [[["new"], ["lookup", a], ["insert_back", a], ["insert_back", b], ["insert_back", c], ["lookup", a], ["lookup", c], ["lookup", z],
        ["delete_node", b], ["lookup", b], ["lookup", c]]],
      print_list: [[["new"], ["print_list"], ["insert_back", a], ["print_list"], ["insert_back", b], ["insert_back", c], ["print_list"], ["delete_head"], ["print_list"]]],
    };
    const groups = [];
    for (const n of names) groups.push(...G[n]);
    if (rnd) groups.push(randomGroup(o, rnd));
    return groups;
  }

  function mulberry32(seed) {
    return () => {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function randomGroup(o, seed) {
    const r = mulberry32(seed);
    const vals = VALUES[o.type];
    const pick = () => vals[Math.floor(r() * vals.length)];
    const ops = [["new"]];
    const A = [];
    for (let i = 0; i < 60; i++) {
      const x = r();
      if (x < 0.25) { const y = pick(); ops.push(["insert_front", y]); A.unshift(y); }
      else if (x < 0.5) { const y = pick(); ops.push(["insert_back", y]); A.push(y); }
      else if (x < 0.62 && A.length) { ops.push(["delete_head"]); A.shift(); }
      else if (x < 0.78) { const y = pick(); ops.push(["delete_node", y]); const k = A.indexOf(y); if (k >= 0) A.splice(k, 1); }
      else if (x < 0.9) ops.push(["lookup", pick()]);
      else if (x < 0.95) ops.push(["print_list"]);
      else ops.push(["is_list"]);
    }
    return ops;
  }

  // ops -> IR statements for the tests' main, and the output a correct list prints
  function compile(groups, o) {
    const v = variant(o);
    const nm = (n) => fname(n, o.lang);
    const lit = (x) => (typeof x === "number" ? String(x) : JSON.stringify(x));
    const label = (x) => String(x);
    const out = [];
    const ir = [];
    const show = (A) => `[${A.join(", ")}] size=${A.length}`;
    const [segFirst, segEnd, segOff] = segment(v);
    const bs = breaks(v);
    for (const g of groups) {
      const body = [];
      let A = [];
      for (const [kind, x] of g) {
        switch (kind) {
          case "node":
            body.push(S("", `let n = new(${lit(x)})`), S("", `print "Node(${label(x)}): data="`), S("", "print n.data"),
              S("", 'print ", next is null: "'), S("", "printb n.next == null"));
            if (v.doubly) body.push(S("", 'print ", prev is null: "'), S("", "printb n.prev == null"));
            body.push(S("", "println"));
            out.push(`Node(${label(x)}): data=${label(x)}, next is null: true${v.doubly ? ", prev is null: true" : ""}`);
            break;
          case "new": A = []; body.push(S("", "list L"), S("", 'print "new list: "'), S("", "show(L)")); out.push(`new list: ${show(A)}`); break;
          case "fill": for (const y of x) { body.push(S("", `ref_insert_back(L, ${lit(y)})`)); A.push(y); } break;
          case "insert_front": case "insert_back":
            body.push(S("", `print "${nm(kind)}(L, ${label(x)}): "`), S("", `${kind}(L, ${lit(x)})`), S("", "show(L)"));
            kind === "insert_front" ? A.unshift(x) : A.push(x);
            out.push(`${nm(kind)}(L, ${label(x)}): ${show(A)}`);
            break;
          case "delete_head": {
            body.push(S("", `print "${nm("delete_head")}(L) = "`), S("", "print delete_head(L)"), S("", 'print ": "'), S("", "show(L)"));
            const y = A.shift();
            out.push(`${nm("delete_head")}(L) = ${label(y)}: ${show(A)}`);
            break;
          }
          case "delete_node": {
            body.push(S("", `print "${nm("delete_node")}(L, ${label(x)}) = "`), S("", `printb delete_node(L, ${lit(x)})`), S("", 'print ": "'), S("", "show(L)"));
            const i = A.indexOf(x);
            if (i >= 0) A.splice(i, 1);
            out.push(`${nm("delete_node")}(L, ${label(x)}) = ${i >= 0}: ${show(A)}`);
            break;
          }
          case "lookup":
            body.push(S("", `print "${nm("lookup")}(L, ${label(x)}) = "`), S("", `printb lookup(L, ${lit(x)})`), S("", "println"));
            out.push(`${nm("lookup")}(L, ${label(x)}) = ${A.includes(x)}`);
            break;
          case "is_empty": case "is_list":
            body.push(S("", `print "${nm(kind)}(L) = "`), S("", `printb ${kind}(L)`), S("", "println"));
            out.push(`${nm(kind)}(L) = ${kind === "is_empty" ? A.length === 0 : true}`);
            break;
          case "print_list":
            body.push(S("", `print "${nm("print_list")}(L) prints: "`), S("", "print_list(L)"));
            out.push(`${nm("print_list")}(L) prints: [${A.join(", ")}]`);
            break;
          case "break": {
            const [stm, desc] = bs[x];
            body.push(S("", stm), S("", `print "after ${desc}: ${nm("is_list")}(L) = "`), S("", "printb is_list(L)"), S("", "println"));
            out.push(`after ${desc}: ${nm("is_list")}(L) = false`);
            break;
          }
          case "segment": {
            // "ring": once more around the circle, which comes back to `end` (wrong: it passes `end` on the way)
            const nExpr = x === "ring" ? (v.dummy ? "L.size + L.size + 1" : "L.size - 1 + L.size")
              : `L.size${segOff + x === 0 ? "" : segOff + x > 0 ? ` + ${segOff + x}` : ` - ${-(segOff + x)}`}`;
            body.push(S("", `print "${nm("is_segment")}(${segFirst}, ${segEnd}, ${nExpr}) = "`), S("", `printb is_segment(${segFirst}, ${segEnd}, ${nExpr})`), S("", "println"));
            out.push(`${nm("is_segment")}(${segFirst}, ${segEnd}, ${nExpr}) = ${x === 0}`);
            break;
          }
        }
      }
      ir.push(S("", "block", body));
    }
    return { ir, expected: out.join("\n") + "\n" };
  }

  // The test program for steps [0, upto) (or the final challenge): returns the pieces DSARun.assemble puts around the
  // user's code, and the expected output.
  function harness(o, upto, opts = {}) {
    const st = steps(o);
    const names = opts.final ? st[st.length - 1].tests : st[upto - 1].tests;
    const groups = testGroups(names, o, opts.final ? (opts.seed || 1) : 0);
    const { ir, expected } = compile(groups, o);
    const v = variant(o);
    const L = o.lang;
    const mainT = o.type === "int" ? { py: "int", cpp: "int", java: "int", rs: "i32" }[L] : { py: "str", cpp: "std::string", java: "String", rs: "String" }[L];
    const refCtx = context(o, { ref: true, qual: true });
    const pieces = [];
    if (names.includes("list")) {
      // the harness's own helpers, built on the user's Node and LinkedList types
      const ref = [{ name: "is_segment", steps: isSegment(v) }, { name: "insert_back", steps: insertBack(v) }, { name: "show", steps: showFn(v) }];
      for (const f of ref) pieces.push(DL.renderFunction(f, refCtx).code);
    }
    const mctx = context(o, { main: true, qual: true, T: mainT, vars: {} });
    const mainLines = DL.renderLines(ir, mctx);
    return { helpers: pieces.join("\n\n"), main: mainLines.join("\n"), expected };
  }

  // the final challenge's problem statement (markdown)
  function problem(o) {
    const v = variant(o);
    const c = context(o);
    const fs = fields(v);
    const L = o.lang;
    // the exact headers the tutorial uses (Rust's generic bounds included)
    const heads = {};
    steps(o).forEach((st) => (st.fns || []).forEach((f) => { heads[f.name] = DL.renderFunction(f, c).header; }));
    const sig = (n) => heads[n].filter((l) => !l.startsWith("template")).join(" ").replace(/\s*[{:]$/, "");
    const lines = [];
    lines.push(`Build a **${describe(o)}** from a blank file, with no hints: everything the tutorial walked you through.`);
    lines.push("");
    lines.push(`**Node.** Fields \`data\`, \`next\`${v.doubly ? ", `prev`" : ""}; ${L === "rs" ? "`Node::new(data)` returns a raw pointer to a new heap node" : "a constructor taking the element"}, with null pointers.`);
    lines.push("");
    lines.push(`**LinkedList.** Fields ${fs.map((f) => `\`${f}\``).join(", ")} and \`size\`. ${{ py: "`LinkedList()`", cpp: "`LinkedList L;`", java: "`new LinkedList()`", rs: "`LinkedList::new()`" }[L]} makes an empty list.` +
      (v.dummy ? (v.circular ? " An empty list is one dummy node pointing to itself." : " An empty list is one dummy node, which is also the head.") : ""));
    lines.push("");
    lines.push("**Functions** (exact names and signatures):");
    lines.push("");
    const what = {
      is_segment: "true if following next from start reaches end after exactly n nodes",
      is_list: "the representation invariant",
      is_empty: "true if the list has no elements",
      insert_front: "add x before the first element, O(1)",
      insert_back: "add x after the last element, O(1)",
      delete_head: "remove and return the first element (requires a non-empty list), O(1)",
      delete_node: "remove the first node holding x; false if there is none",
      lookup: "true if some node holds x",
      print_list: "print the elements like `[3, 5, 7]` (or `[]`) and end the line",
    };
    for (const n of Object.keys(what)) lines.push(`* \`${sig(n)}\`: ${what[n]}`);
    lines.push("");
    lines.push(`Check \`${fname("is_list", L)}(L)\` at the start and end of every operation. ${L === "java" ? "Put the functions inside `class LinkedList` as static methods. " : ""}You don't need a \`main\`: the checker adds its own (yours, if any, is ignored) and runs every test from the tutorial plus a random sequence of operations.`);
    return lines.join("\n");
  }

  const tutorial = {
    id: "linked-lists", num: "13.01", title: "Linked lists", heading: "Build a linked list, step by step", OPTIONS, DEFAULTS, describe, complexity, diagram, render, program, harness, problem,
    stepCount: 8, variant, fields, breaks, steps, context, FNS, _internals: { isSegment, isList, insertFront, insertBack, deleteHead, deleteNode, lookup, printList, showFn, testGroups, compile },
  };
  root.DSATutorials = root.DSATutorials || {};
  root.DSATutorials["linked-lists"] = tutorial;
  if (typeof module !== "undefined" && module.exports) module.exports = tutorial;
})(typeof window !== "undefined" ? window : globalThis);
