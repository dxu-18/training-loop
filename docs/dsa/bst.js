// 13.09 Binary search trees: a plain BST, an AVL tree or a splay tree, in Python, C++, Java or Rust. Node functions are
// recursive and return the new root of their subtree (B.root = node_insert(B.root, x)).
(function (root) {
  "use strict";
  const DL = root.DSALang || (typeof require !== "undefined" ? require("./lang.js") : null);
  const C = root.DSACommon || (typeof require !== "undefined" ? require("./common.js") : null);
  const { S, fname } = C;

  const OPTIONS = [
    { id: "lang", label: "Language", choices: [["py", "Python"], ["cpp", "C++"], ["java", "Java"], ["rs", "Rust"]] },
    { id: "type", label: "Elements", choices: [["int", "int"], ["generic", "generic (T)"]],
      help: { int: "Every element is an integer.", generic: "Any element type that can be ordered: a type parameter T (Comparable in Java, PartialOrd in Rust)." } },
    { id: "tree", label: "Tree", choices: [["plain", "plain BST"], ["avl", "AVL"], ["splay", "splay"]],
      help: { plain: "Smaller elements to the left, bigger to the right, nothing else. Fast on random input; a sorted input makes it a chain.",
        avl: "A BST that stores each node's height and rotates whenever the two sides differ by more than 1: always O(log n) high.",
        splay: "A BST that moves every node it looks at to the root with rotations (splaying). No balance information, O(log n) amortized, and recently used elements are fast." } },
  ];
  const DEFAULTS = { lang: "py", type: "int", tree: "plain" };
  const avl = (o) => o.tree === "avl", splay = (o) => o.tree === "splay";
  const describe = (o) => `${{ plain: "binary search tree", avl: "AVL tree", splay: "splay tree" }[o.tree]} of ${o.type === "int" ? "int" : "generic"} elements, in ${DL.LANGS[o.lang].name}`;
  const ctxExtra = () => ({ node: { name: "Node", fields: { data: "T", left: "N", right: "N", height: "int" } } });

  function complexity(o) {
    const t = o.tree;
    const op = { plain: ["O(h)", "h = log n on random input, n on sorted input"], avl: ["O(log n)", "the height stays under 1.44 log₂ n"],
      splay: ["O(log n)", "amortized: over a sequence of operations"] }[t];
    return [
      { id: "lookup", label: "Lookup", big: op[0], note: op[1] },
      { id: "insert", label: "Insert", big: op[0], note: t === "avl" ? "plus at most a couple of rotations on the way up" : t === "splay" ? "splay, then split at the root" : "walk down to an empty spot" },
      { id: "delete", label: "Delete", big: op[0], note: t === "splay" ? "splay, then join the two subtrees" : "two children: take the right subtree's minimum" },
      { id: "worst", label: "One operation, worst case", big: t === "avl" ? "O(log n)" : "O(n)", note: t === "plain" ? "a chain: sorted input" : t === "splay" ? "a single splay can be long; the next ones pay less" : "the height is always O(log n)" },
      { id: "height", label: "Height", big: t === "avl" ? "O(log n)" : "O(n)", note: t === "avl" ? "kept balanced" : "nothing keeps it balanced" },
      { id: "min", label: "Minimum", big: op[0], note: "keep going left" },
      { id: "inorder", label: "In-order traversal", big: "O(n)", note: "left subtree, node, right subtree: sorted order" },
      { id: "space", label: "Space", big: "O(n)", note: t === "avl" ? "n nodes, each with a height" : "n nodes" },
      { id: "hash", label: "Hash table, for comparison", big: "O(1)", note: "expected, but no order: no minimum or sorted traversal", extra: true },
      { id: "is_bst", label: "Contract is_bst", big: "O(n)", note: "checks every node against bounds from its ancestors" },
    ];
  }

  // ---------------------------------------------------------------- the model: the exact tree every test expects
  const lt = (a, b) => a < b;
  function model(o) {
    const t = o.tree;
    const st = { root: null, size: 0 };
    const N = (x) => ({ d: x, l: null, r: null, h: 1 });
    const H = (n) => (n ? n.h : 0);
    const fix = (n) => { const a = H(n.l), b = H(n.r); n.h = (a > b ? a : b) + 1; };
    const rotR = (n) => { const l = n.l; n.l = l.r; l.r = n; if (t === "avl") { fix(n); fix(l); } return l; };
    const rotL = (n) => { const r = n.r; n.r = r.l; r.l = n; if (t === "avl") { fix(n); fix(r); } return r; };
    const rebalance = (n) => {
      fix(n);
      if (H(n.l) > H(n.r) + 1) { if (H(n.l.l) < H(n.l.r)) n.l = rotL(n.l); return rotR(n); }
      if (H(n.r) > H(n.l) + 1) { if (H(n.r.r) < H(n.r.l)) n.r = rotR(n.r); return rotL(n); }
      return n;
    };
    const fin = (n) => (t === "avl" ? rebalance(n) : n);
    const ins = (n, x) => { if (!n) return N(x); if (lt(x, n.d)) n.l = ins(n.l, x); else if (lt(n.d, x)) n.r = ins(n.r, x); return fin(n); };
    const minN = (n) => { while (n.l) n = n.l; return n; };
    const delMin = (n) => { if (!n.l) return n.r; n.l = delMin(n.l); return fin(n); };
    const del = (n, x) => {
      if (lt(x, n.d)) n.l = del(n.l, x);
      else if (lt(n.d, x)) n.r = del(n.r, x);
      else { if (!n.l) return n.r; if (!n.r) return n.l; n.d = minN(n.r).d; n.r = delMin(n.r); }
      return fin(n);
    };
    const look = (n, x) => (!n ? false : x === n.d ? true : lt(x, n.d) ? look(n.l, x) : look(n.r, x));
    const sp = (n, x) => {
      if (!n || n.d === x) return n;
      if (lt(x, n.d)) {
        if (!n.l) return n;
        if (lt(x, n.l.d)) { n.l.l = sp(n.l.l, x); const m = rotR(n); return m.l ? rotR(m) : m; }
        if (lt(n.l.d, x)) { n.l.r = sp(n.l.r, x); if (n.l.r) n.l = rotL(n.l); }
        return n.l ? rotR(n) : n;
      }
      if (!n.r) return n;
      if (lt(n.r.d, x)) { n.r.r = sp(n.r.r, x); const m = rotL(n); return m.r ? rotL(m) : m; }
      if (lt(x, n.r.d)) { n.r.l = sp(n.r.l, x); if (n.r.l) n.r = rotR(n.r); }
      return n.r ? rotL(n) : n;
    };
    const ordered = (n, lo, hi) => !n || ((!lo || lt(lo.d, n.d)) && (!hi || lt(n.d, hi.d)) && ordered(n.l, lo, n) && ordered(n.r, n, hi));
    const count = (n) => (n ? 1 + count(n.l) + count(n.r) : 0);
    const isAvl = (n) => !n || (H(n.l) <= H(n.r) + 1 && H(n.r) <= H(n.l) + 1 && n.h === (H(n.l) > H(n.r) ? H(n.l) : H(n.r)) + 1 && isAvl(n.l) && isAvl(n.r));
    const lab = (n) => (t === "avl" ? `${n.d}:${n.h}` : String(n.d));   // (AVL: each node's height too)
    const shapeOf = (n) => (!n ? "_" : !n.l && !n.r ? lab(n) : `${lab(n)}(${shapeOf(n.l)}, ${shapeOf(n.r)})`);
    const inorder = (n, out = []) => { if (n) { inorder(n.l, out); out.push(n.d); inorder(n.r, out); } return out; };
    return {
      st,
      insert: (x) => {
        if (t === "splay") {
          if (!st.root) { st.root = N(x); st.size += 1; return; }
          st.root = sp(st.root, x);
          if (st.root.d === x) return;
          const n = N(x);
          if (lt(x, st.root.d)) { n.l = st.root.l; st.root.l = null; n.r = st.root; } else { n.r = st.root.r; st.root.r = null; n.l = st.root; }
          st.root = n; st.size += 1; return;
        }
        if (look(st.root, x)) return;
        st.root = ins(st.root, x); st.size += 1;
      },
      lookup: (x) => { if (t !== "splay") return look(st.root, x); if (!st.root) return false; st.root = sp(st.root, x); return st.root.d === x; },
      remove: (x) => {
        if (t === "splay") {
          if (!st.root) return false;
          st.root = sp(st.root, x);
          if (st.root.d !== x) return false;
          const l = st.root.l, r = st.root.r;
          if (!l) st.root = r; else { const m = sp(l, x); m.r = r; st.root = m; }
          st.size -= 1; return true;
        }
        if (!look(st.root, x)) return false;
        st.root = del(st.root, x); st.size -= 1; return true;
      },
      rotateRoot: (dir) => { st.root = dir === "right" ? rotR(st.root) : rotL(st.root); },
      rebalanceRoot: () => { st.root = rebalance(st.root); },
      splayRoot: (x) => { st.root = sp(st.root, x); },
      // a small tree built by hand: [element, height, left, right]
      build: (spec) => { const mk = (q) => (q ? { d: q[0], h: q[1], l: mk(q[2]), r: mk(q[3]) } : null); st.root = mk(spec); st.size = 3; },
      min: () => minN(st.root).d,
      inorder: () => inorder(st.root),
      ok: () => ordered(st.root, null, null) && count(st.root) === st.size && (t !== "avl" || isAvl(st.root)),
      show: () => `${shapeOf(st.root)} size=${st.size}`,
      // every node with the path to it from the root ("B.root.left.right")
      nodes: () => { const out = []; const walk = (n, path) => { if (n) { out.push([n, path]); walk(n.l, `${path}.left`); walk(n.r, `${path}.right`); } }; walk(st.root, "B.root"); return out; },
      shape: () => shapeOf(st.root),
    };
  }

  function diagram(o) {
    const m = model(o);
    const v = o.type === "int" ? [40, 20, 60, 10, 30, 50, 70] : ["eel", "cat", "gnu", "ant", "dog", "fox", "yak"];
    (avl(o) ? [...v].sort((a, b) => (a < b ? -1 : 1)) : v).forEach((x) => m.insert(x));
    if (splay(o)) m.lookup(v[3]);
    // the classic drawing: each node in the column of its in-order position, one row per depth
    const order = [];
    const walk = (n, d) => { if (n) { walk(n.l, d + 1); order.push([n, d]); walk(n.r, d + 1); } };
    walk(m.st.root, 0);
    const depth = Math.max(...order.map(([, d]) => d));
    const rows = [];
    for (let d = 0; d <= depth; d++) {
      rows.push([{ label: d === 0 ? "B.root" : `depth ${d}` }, ...order.map(([n, nd]) => (nd === d ? { box: String(n.d), tags: [] } : { box: String(n.d), tags: [], cls: "spacer" }))]);
    }
    return { rows, caption: {
      plain: "Everything in a node's left subtree is smaller than it and everything in its right subtree is bigger. Lookup goes left or right at each node: O(height).",
      avl: "Inserting the elements in sorted order would make a plain BST a chain; the AVL tree's rotations keep it perfectly balanced here.",
      splay: `After looking up ${v[3]}, splaying has moved it to the root. Elements used recently stay near the top.`,
    }[o.tree] };
  }

  // ---------------------------------------------------------------- specs
  const O = "BST", M = "mut:BST", R = "ref:BST", ORD = ["PartialOrd"];
  function FNS(o) {
    const f = {
      is_ordered: { params: [["n", "N"], ["lo", "N"], ["hi", "N"]], ret: "bool", owner: O, bounds: ORD, contract: true },
      node_size: { params: [["n", "N"]], ret: "int", owner: O, contract: true },
      is_bst: { params: [["B", R]], ret: "bool", owner: O, bounds: ORD, contract: true },
      bst_size: { params: [["B", R]], ret: "int", owner: O },
      rotate_right: { params: [["n", "N"]], ret: "N", owner: O },
      rotate_left: { params: [["n", "N"]], ret: "N", owner: O },
      node_lookup: { params: [["n", "N"], ["x", "Tref"]], ret: "bool", owner: O, bounds: ORD },
      bst_lookup: { params: [["B", splay(o) ? M : R], ["x", "Tref"]], ret: "bool", owner: O, bounds: ORD },
      node_insert: { params: [["n", "N"], ["x", "T"]], ret: "N", owner: O, bounds: ORD },
      bst_insert: { params: [["B", M], ["x", "T"]], ret: "void", owner: O, bounds: ORD },
      node_min: { params: [["n", "N"]], ret: "N", owner: O },
      bst_min: { params: [["B", R]], ret: "Tpeek", owner: O },
      node_print: { params: [["n", "N"], ["k", "int"]], ret: "int", owner: O, bounds: ["std::fmt::Display"] },
      bst_print: { params: [["B", R]], ret: "void", owner: O, bounds: ["std::fmt::Display"] },
      node_delete_min: { params: [["n", "N"]], ret: "N", owner: O },
      node_delete: { params: [["n", "N"], ["x", "Tref"]], ret: "N", owner: O, bounds: ["PartialOrd", "Clone"] },
      bst_delete: { params: [["B", M], ["x", "Tref"]], ret: "bool", owner: O, bounds: ["PartialOrd", "Clone"] },
      splay: { params: [["n", "N"], ["x", "Tref"]], ret: "N", owner: O, bounds: ORD },
      height: { params: [["n", "N"]], ret: "int", owner: O },
      fix_height: { params: [["n", "N"]], ret: "void", owner: O },
      rebalance: { params: [["n", "N"]], ret: "N", owner: O },
      is_avl: { params: [["n", "N"]], ret: "bool", owner: O, contract: true },
      show_node: { params: [["n", "N"], ["depth", "int"]], ret: "void", harness: true, bounds: ["std::fmt::Display"] },
      show: { params: [["B", R]], ret: "void", harness: true, bounds: ["std::fmt::Display", "PartialOrd"] },
      shape: { params: [["B", R]], ret: "void", harness: true, bounds: ["std::fmt::Display"] },
    };
    if (splay(o)) { f.bst_delete.bounds = ORD; f.bst_insert.bounds = ORD; }
    return f;
  }
  const STRUCTS = { BST: { fields: { root: "N", size: "int" } } };

  // ---------------------------------------------------------------- the functions
  const REQ = S("**Requires**: the tree is valid when the function starts.", "assert is_bst(B)");
  const ENS = S("**Ensures**: still valid when it returns.", "assert is_bst(B)");
  const isOrdered = () => [
    S("An empty subtree is ordered.", "if n == null", [S("", "return true")]),
    S("`lo` and `hi` are nodes bounding this subtree from its ancestors ({null} = no bound): the element must be strictly between them.", "if lo != null and n.data <= lo.data", [S("", "return false")]),
    S("", "if hi != null and n.data >= hi.data", [S("", "return false")]),
    S("Everything on the left must be smaller than `n`, everything on the right bigger.", "return is_ordered(n.left, lo, n) and is_ordered(n.right, n, hi)")];
  const nodeSize = () => [S("", "if n == null", [S("", "return 0")]), S("A node plus both subtrees.", "return 1 + node_size(n.left) + node_size(n.right)")];
  const isAvl = () => [
    S("", "if n == null", [S("", "return true")]),
    S("The two subtrees' heights…", "let hl = height(n.left)"), S("", "let hr = height(n.right)"),
    S("…differ by at most 1 (balanced)…", "if hl > hr + 1 or hr > hl + 1", [S("", "return false")]),
    S("…and the stored height is 1 more than the taller one.", "if hl > hr", [S("", "if n.height != hl + 1", [S("", "return false")])]),
    S("", "elif n.height != hr + 1", [S("", "return false")]),
    S("Both subtrees are AVL too.", "return is_avl(n.left) and is_avl(n.right)")];
  const isBst = (o) => () => [S("Ordered, with exactly `size` nodes" + (avl(o) ? ", and balanced with correct heights." : "."),
    avl(o) ? "return is_ordered(B.root, null, null) and node_size(B.root) == B.size and is_avl(B.root)" : "return is_ordered(B.root, null, null) and node_size(B.root) == B.size")];
  const height = () => [S("An empty subtree has height 0…", "if n == null", [S("", "return 0")]), S("…a node knows its own.", "return n.height")];
  const fixHeight = () => [
    S("1 more than the taller subtree.", "let hl = height(n.left)"), S("", "let hr = height(n.right)"),
    S("", "if hl > hr", [S("", "n.height = hl + 1")]), S("", "else", [S("", "n.height = hr + 1")])];
  const rot = (o, dir) => () => {
    const [a, b] = dir === "right" ? ["left", "right"] : ["right", "left"];
    return [
      S(`\`${a === "left" ? "l" : "r"}\`, ${a === "left" ? "the left" : "the right"} child, will be the new root of this subtree.`, `let c = n.${a}`),
      S(`Its ${b} subtree moves over to become \`n\`'s ${a} subtree (everything in it is between \`c\` and \`n\`).`, `n.${a} = c.${b}`),
      S(`\`n\` becomes \`c\`'s ${b} child.`, `c.${b} = n`),
      ...(avl(o) ? [S("`n` is lower now and `c` higher: fix their heights, `n` first (it's below `c`).", "fix_height(n)"), S("", "fix_height(c)")] : []),
      S("Return the new root of the subtree.", "return c")];
  };
  const rebalance = () => [
    S("First the node's own height.", "fix_height(n)"),
    S("Left side taller by 2:", "if height(n.left) > height(n.right) + 1", [
      S("if the extra height is in the left child's *right* subtree (left-right case), rotate the left child left first…", "if height(n.left.left) < height(n.left.right)", [S("", "n.left = rotate_left(n.left)")]),
      S("…then rotate right.", "return rotate_right(n)")]),
    S("Right side taller by 2: the mirror image.", "if height(n.right) > height(n.left) + 1", [
      S("", "if height(n.right.right) < height(n.right.left)", [S("", "n.right = rotate_right(n.right)")]),
      S("", "return rotate_left(n)")]),
    S("Balanced already.", "return n")];
  const splayFn = () => [
    S("Nothing to do: an empty tree, or `x` is already at the top.", "if n == null or n.data == x", [S("", "return n")]),
    S("`x` is on the left:", "if x < n.data", [
      S("", "if n.left == null", [S("", "return n")]),
      S("**Zig-zig** (left of the left child): splay it up the left-left subtree, then rotate right twice.", "if x < n.left.data", [
        S("", "n.left.left = splay(n.left.left, x)"), S("", "let m = rotate_right(n)"),
        S("", "if m.left == null", [S("", "return m")]), S("", "return rotate_right(m)")]),
      S("**Zig-zag** (right of the left child): splay it up the left-right subtree, rotate the left child left…", "if n.left.data < x", [
        S("", "n.left.right = splay(n.left.right, x)"), S("", "if n.left.right != null", [S("", "n.left = rotate_left(n.left)")])]),
      S("…then (also for plain **zig**) rotate right once.", "return rotate_right(n)")]),
    S("`x` is on the right: the mirror image.", "if n.right == null", [S("", "return n")]),
    S("", "if n.right.data < x", [S("", "n.right.right = splay(n.right.right, x)"), S("", "let m = rotate_left(n)"),
      S("", "if m.right == null", [S("", "return m")]), S("", "return rotate_left(m)")]),
    S("", "if x < n.right.data", [S("", "n.right.left = splay(n.right.left, x)"), S("", "if n.right.left != null", [S("", "n.right = rotate_right(n.right)")])]),
    S("", "return rotate_left(n)")];
  const nodeLookup = () => [
    S("Fell off the tree: not there.", "if n == null", [S("", "return false")]),
    S("Found it.", "if x == n.data", [S("", "return true")]),
    S("Smaller: it can only be on the left…", "if x < n.data", [S("", "return node_lookup(n.left, x)")]),
    S("…otherwise only on the right.", "return node_lookup(n.right, x)")];
  const lookup = (o) => () => (splay(o) ? [
    S("An empty tree holds nothing.", "if B.root == null", [S("", "return false")]),
    S("Splay `x` (or the last node on its path) to the root…", "B.root = splay(B.root, x)"),
    S("…and look there.", "return B.root.data == x")]
    : [S("Search from the root.", "return node_lookup(B.root, x)")]);
  const nodeInsert = (o) => () => [
    S("An empty spot: the new node goes here.", "if n == null", [S("", "return new(x)")]),
    S("Smaller: insert into the left subtree (its root may change)…", "if x < n.data", [S("", "n.left = node_insert(n.left, x)")]),
    S("…bigger: into the right.", "elif n.data < x", [S("", "n.right = node_insert(n.right, x)")]),
    avl(o) ? S("Rebalance on the way back up.", "return rebalance(n)") : S("This subtree's root doesn't change.", "return n")];
  const insert = (o) => () => (splay(o) ? [REQ,
    S("An empty tree: the new node is the root.", "if B.root == null", [S("", "B.root = new(x)"), S("", "B.size += 1"),
      { ...S("", "return"), noMutate: "without it, the duplicate check just below returns anyway" }]),
    S("Splay `x`'s spot to the root.", "B.root = splay(B.root, x)"),
    S("Already there: nothing to add.", "if B.root.data == x", [S("", "return")]),
    S("The new node becomes the root, splitting the old root's tree around `x`.", "let n = new(x)"),
    S("`x` is smaller than the root (compare through `n`: in Rust, `x` now belongs to the node): the root's left subtree goes left of `n`, and the root (with its right subtree) goes right.", "if n.data < B.root.data", [
      S("", "n.left = B.root.left"), S("", "B.root.left = null"), S("", "n.right = B.root")]),
    S("Otherwise the mirror image.", "else", [S("", "n.right = B.root.right"), S("", "B.root.right = null"), S("", "n.left = B.root")]),
    S("", "B.root = n"), S("One more element.", "B.size += 1"), ENS]
    : [REQ,
      S("Already in the tree: nothing to add.", "if bst_lookup(B, x)", [S("", "return")]),
      S("Insert from the root (the root itself may change).", "B.root = node_insert(B.root, x)"),
      S("One more element.", "B.size += 1"), ENS]);
  const nodeMin = () => [S("Keep going left: the leftmost node holds the smallest element.", "let p = n"), S("", "while p.left != null", [S("", "p = p.left")]), S("", "return p")];
  const bstMin = () => [S("**Requires**: a non-empty tree.", "assert B.root != null"),
    S((c) => (c.lang === "rs" && c.generic ? "The leftmost node's element (a reference `&T`)." : "The leftmost node's element."), "return node_min(B.root).data")];
  const nodePrint = () => [
    S("`k` elements were printed before this subtree.", "if n == null", [S("", "return k")]),
    S("In order: first the left subtree…", "let c = node_print(n.left, k)"),
    S("…then this node (after a comma, unless it's the first)…", "if c > 0", [S("", 'print ", "')]), S("", "print n.data"),
    S("…then the right subtree. Return how many are printed now.", "return node_print(n.right, c + 1)")];
  const bstPrint = () => [S("", 'print "["'), S("Print the tree in order: sorted.", "node_print(B.root, 0)"), S("", 'println "]"')];
  const nodeDeleteMin = (o) => () => [
    S("No left child: this is the minimum; its right subtree takes its place.", "if n.left == null", [S("", "let r = n.right"), S("", "free n"), S("", "return r")]),
    S("Otherwise delete the minimum from the left subtree.", "n.left = node_delete_min(n.left)"),
    avl(o) ? S("", "return rebalance(n)") : S("", "return n")];
  const nodeDelete = (o) => () => [
    S("Smaller: delete from the left subtree…", "if x < n.data", [S("", "n.left = node_delete(n.left, x)")]),
    S("…bigger: from the right.", "elif n.data < x", [S("", "n.right = node_delete(n.right, x)")]),
    S("Found `x`'s node:", "else", [
      // (in an AVL tree a node without a left child has at most a leaf on its right: the general case below gives the same tree)
      { ...S("no left child: its right subtree takes its place…", "if n.left == null", [S("", "let r = n.right"), S("", "free n"),
        { ...S("", "return r"), noMutate: avl(o) && "AVL: equivalent to the two-children case" }]), noMutate: avl(o) && "AVL: equivalent to the two-children case" },
      S("…no right child: its left subtree does…", "if n.right == null", [S("", "let l = n.left"), S("", "free n"), S("", "return l")]),
      S("…two children: copy in the smallest element of the right subtree (the next one in order), and delete that node instead.", "let m = node_min(n.right)"),
      S("", "n.data = m.data"), S("", "n.right = node_delete_min(n.right)")]),
    avl(o) ? S("Rebalance on the way back up.", "return rebalance(n)") : S("", "return n")];
  const del = (o) => () => (splay(o) ? [REQ,
    S("An empty tree holds nothing.", "if B.root == null", [S("", "return false")]),
    S("Splay `x` to the root.", "B.root = splay(B.root, x)"),
    S("Not there.", "if B.root.data != x", [S("", "return false")]),
    S("Remove the root, keeping its two subtrees.", "let l = B.root.left"), S("", "let r = B.root.right"), S("", "free B.root"),
    S("No left subtree: the right one is the tree.", "if l == null", [S("", "B.root = r")]),
    S("Otherwise splay the left subtree's maximum to its top (everything there is smaller than `x`), so it has no right child, and hang the right subtree there.", "else", [
      S("", "let m = splay(l, x)"), S("", "m.right = r"), S("", "B.root = m")]),
    S("One element fewer.", "B.size -= 1"), ENS, S("", "return true")]
    : [REQ,
      S("Not in the tree: nothing to delete.", "if not bst_lookup(B, x)", [S("", "return false")]),
      S("Delete from the root.", "B.root = node_delete(B.root, x)"),
      S("One element fewer.", "B.size -= 1"), ENS, S("", "return true")]);

  // ---------------------------------------------------------------- declarations, per language
  function decl(c, o) {
    const g = c.generic, L = c.lang, T = c.T, A = avl(o);
    if (L === "py") {
      const hint = g ? "Node[T] | None" : "Node | None";
      return [
        { say: g ? "Declare the node class `Node[T]`." : "Declare the node class `Node`.", code: [g ? "class Node[T]:" : "class Node:"], body: [
          { say: "Its constructor takes the element; both children start empty.", code: [`def __init__(self, data: ${T}) -> None:`], body: [
            { say: "", code: ["self.data = data"] }, { say: "", code: [`self.left: ${hint} = None`] }, { say: "", code: [`self.right: ${hint} = None`] },
            ...(A ? [{ say: "AVL: the height of this node's subtree; a new node is a leaf, height 1.", code: ["self.height = 1"] }] : [])] }] },
        { say: "The tree: its root and its size.", code: [g ? "class BST[T]:" : "class BST:"], body: [{ say: "", code: ["def __init__(self) -> None:"], body: [
          { say: "", code: [`self.root: ${hint} = None`] }, { say: "", code: ["self.size = 0"] }] }] },
      ];
    }
    if (L === "cpp") {
      const P = g ? "Node<T>*" : "Node*";
      return [
        { say: "At the top of the file: `<iostream>` for printing, `<cassert>` for contracts.", code: ["#include <iostream>", "#include <cassert>"] },
        { say: g ? "Declare the node struct, a template over `T`." : "Declare the node struct.", code: g ? ["template <typename T>", "struct Node {"] : ["struct Node {"], close: ["};"], body: [
          { say: "", code: [`${T} data;`] }, { say: "Both children start empty.", code: [`${P} left = nullptr;`] }, { say: "", code: [`${P} right = nullptr;`] },
          ...(A ? [{ say: "AVL: the height of this node's subtree; a new node is a leaf, height 1.", code: ["int height = 1;"] }] : []),
          { say: "", code: [`Node(${T} data) : data(data) {}`] }] },
        { say: "The tree: its root and its size. The defaults make an empty tree.", code: g ? ["template <typename T>", "struct BST {"] : ["struct BST {"], close: ["};"], body: [
          { say: "", code: [`${P} root = nullptr;`] }, { say: "", code: ["int size = 0;"] }] },
      ];
    }
    if (L === "java") {
      const P = g ? "Node<T>" : "Node";
      return [
        { say: g ? "Declare the class `Node<T>`." : "Declare the class `Node`.", code: [g ? "class Node<T> {" : "class Node {"], close: ["}"], body: [
          { say: "", code: [`${T} data;`] }, { say: "Both children start as `null`.", code: [`${P} left;`] }, { say: "", code: [`${P} right;`] },
          ...(A ? [{ say: "AVL: the height of this node's subtree; a new node is a leaf, height 1.", code: ["int height = 1;"] }] : []),
          { say: "", code: [`Node(${T} data) {`], close: ["}"], body: [{ say: "", code: ["this.data = data;"] }] }] },
        { say: "The tree: its root and its size. Every function goes inside this class, as a `static` method.", code: [g ? "class BST<T> {" : "class BST {"], close: ["}"],
          deferClose: true, cls: O, body: [{ say: "", code: [`${P} root;`] }, { say: "", code: ["int size;"] }] },
      ];
    }
    const N = g ? "Node<T>" : "Node";
    return [
      { say: "At the very top: allow capital variable names (the tree is called `B`), and import `std::ptr` for null pointers.", code: ["#![allow(non_snake_case)]", "use std::ptr;"] },
      { say: "Declare the node. Its children are raw pointers, like the linked lists' nodes: rotations rewire them freely.", code: [`struct ${N} {`], close: ["}"], body: [
        { say: "", code: [`data: ${T},`] }, { say: "", code: [`left: *mut ${N},`] }, { say: "", code: [`right: *mut ${N},`] },
        ...(A ? [{ say: "AVL: the height of this node's subtree.", code: ["height: usize,"] }] : [])] },
      { say: "`Node::new` puts a node on the heap and returns a raw pointer to it." + (A ? " A new node is a leaf, height 1." : ""), code: [g ? "impl<T> Node<T> {" : "impl Node {"], close: ["}"], body: [
        { say: "", code: [`fn new(data: ${T}) -> *mut ${N} {`], close: ["}"], body: [
          { say: "", code: [`Box::into_raw(Box::new(Node { data, left: ptr::null_mut(), right: ptr::null_mut()${A ? ", height: 1" : ""} }))`] }] }] },
      { say: "The tree: its root and its size.", code: [g ? "struct BST<T> {" : "struct BST {"], close: ["}"], body: [{ say: "", code: [`root: *mut ${N},`] }, { say: "", code: ["size: usize,"] }] },
      { say: "`BST::new()`: an empty tree.", code: [g ? "impl<T> BST<T> {" : "impl BST {"], close: ["}"], body: [
        { say: "", code: ["fn new() -> Self {"], close: ["}"], body: [{ say: "", code: ["BST { root: ptr::null_mut(), size: 0 }"] }] }] },
    ];
  }

  function steps(o) {
    const fn = (name, body, headerSay) => ({ name, steps: body(), headerSay });
    const on = C.contractsOn(o), A = avl(o), SP = splay(o);
    const out = [
      { id: "tree", title: "The node and the tree",
        goal: "A node holds an element and two children, `left` and `right`" + (A ? ", plus the `height` of its subtree" : "") +
          ". The tree `B` keeps its `root` and `size`. In a binary search tree, everything in a node's left subtree is smaller than it and everything in its right subtree is bigger." +
          (A ? " {fn:height(n)} reads a subtree's height (0 when empty) and {fn:fix_height(n)} recomputes a node's from its children." : ""),
        decl: (c) => decl(c, o),
        fns: [fn("bst_size", () => [S("The number of elements.", "return B.size")], "Write {fn:bst_size}."),
          ...(A ? [fn("height", height, "Write {fn:height}."), fn("fix_height", fixHeight, "Write {fn:fix_height}.")] : [])],
        tests: ["tree"] },
      { id: "contracts", title: "Contracts",
        goal: "{fn:is_ordered(n, lo, hi)} checks the ordering rule with bounds from the ancestors (a node's grandchild must respect the grandparent too), {fn:node_size(n)} counts nodes" +
          (A ? ", {fn:is_avl(n)} checks the heights and the balance" : "") + ", and {fn:is_bst(B)} puts them together. Every operation checks {fn:is_bst(B)}.",
        fns: [fn("is_ordered", isOrdered, "Write {fn:is_ordered}."), fn("node_size", nodeSize, "Write {fn:node_size}."),
          ...(A ? [fn("is_avl", isAvl, "Write {fn:is_avl}.")] : []), fn("is_bst", isBst(o), "Write {fn:is_bst}.")],
        contractNote: true, tests: ["tree", "contracts"] },
    ];
    const rest = [];
    if (A || SP) {
      rest.push({ id: "rotations", title: "Rotations",
        goal: "A **rotation** lifts a child above its parent and keeps the order: {fn:rotate_right(n)} makes `n`'s left child the root of the subtree, {fn:rotate_left(n)} its right child. O(1): three pointers change" +
          (A ? ", then the two heights." : "."),
        fns: [fn("rotate_right", rot(o, "right"), "Write {fn:rotate_right}; it returns the subtree's new root."), fn("rotate_left", rot(o, "left"), "Write {fn:rotate_left}, the mirror image.")] });
    }
    if (A) {
      rest.push({ id: "rebalance", title: "Rebalance",
        goal: "{fn:rebalance(n)} fixes `n`'s height and, if one side is 2 taller than the other, rotates: once when the extra height is on the outside (left-left or right-right), twice when it's on the inside (left-right or right-left).",
        fns: [fn("rebalance", rebalance, "Write {fn:rebalance}; it returns the subtree's new root.")] });
    }
    if (SP) {
      rest.push({ id: "splay", title: "Splay",
        goal: "{fn:splay(n, x)} brings `x` to the root of the subtree with rotations (or, if `x` isn't there, the last node on its search path). Two levels at a time: **zig-zig** rotates the grandparent first, **zig-zag** the parent; a final **zig** handles the last level.",
        fns: [fn("splay", splayFn, "Write {fn:splay}; it returns the subtree's new root.")] });
    }
    rest.push(
      { id: "lookup", title: "Lookup",
        goal: SP ? "{fn:bst_lookup(B, x)} splays `x` to the root and checks the root. Even a lookup reshapes a splay tree."
          : "{fn:node_lookup(n, x)} compares `x` with `n`'s element and continues left or right: one path from the root, O(height). {fn:bst_lookup(B, x)} starts at the root.",
        fns: [...(SP ? [] : [fn("node_lookup", nodeLookup, "Write {fn:node_lookup}, recursively.")]), fn("bst_lookup", lookup(o), "Write {fn:bst_lookup}.")] },
      { id: "insert", title: "Insert",
        goal: SP ? "{fn:bst_insert(B, x)} splays `x`'s place to the root, then makes a new root node and splits the old tree around it."
          : "{fn:node_insert(n, x)} walks down like lookup and puts a new node in the empty spot it reaches; it returns the subtree's (possibly new) root" + (A ? ", rebalancing on the way back up." : ".") +
            " {fn:bst_insert(B, x)} does nothing if `x` is already there.",
        fns: [...(SP ? [] : [fn("node_insert", nodeInsert(o), "Write {fn:node_insert}, recursively.")]), fn("bst_insert", insert(o), "Write {fn:bst_insert}.")] },
      { id: "min_print", title: "Minimum and in-order",
        goal: "{fn:node_min(n)} keeps going left; {fn:bst_min(B)} returns the smallest element. {fn:node_print(n, k)} prints a subtree **in order** (left subtree, node, right subtree), which is sorted order, and {fn:bst_print(B)} wraps it in brackets.",
        fns: [fn("node_min", nodeMin, "Write {fn:node_min}."), fn("bst_min", bstMin, "Write {fn:bst_min}."), fn("node_print", nodePrint, "Write {fn:node_print}."), fn("bst_print", bstPrint, "Write {fn:bst_print}.")] },
      { id: "delete", title: "Delete",
        goal: SP ? "{fn:bst_delete(B, x)} splays `x` to the root, removes it, and joins the two subtrees: splaying the left one's maximum to its top leaves it no right child, so the right subtree hangs there."
          : "{fn:node_delete(n, x)} finds `x`'s node. With at most one child, that child takes its place. With two, the node takes the smallest element of its right subtree, and {fn:node_delete_min} removes that node instead." +
            (A ? " Both rebalance on the way back up." : ""),
        fns: [...(SP ? [] : [fn("node_delete_min", nodeDeleteMin(o), "Write {fn:node_delete_min}."), fn("node_delete", nodeDelete(o), "Write {fn:node_delete}.")]), fn("bst_delete", del(o), "Write {fn:bst_delete}.")] });
    const all = [...out, ...rest];
    // every step's tests include the earlier steps'
    const ids = all.map((st) => st.id);
    return all.map((st, i) => ({ ...st, tests: ids.slice(0, i + 1) }));
  }

  // ---------------------------------------------------------------- tests
  const VALS = { int: [50, 30, 70, 20, 40, 60, 80, 10, 35, 65, 25, 45], generic: ["mole", "fox", "seal", "cat", "hen", "pig", "yak", "ant", "gnu", "owl", "dog", "ibis"] };
  const BIG = { int: 99, generic: "zebu" }, SMALL = { int: 1, generic: "aaa" }, ABSENT = { int: [5, 55, 90], generic: ["aardvark", "newt", "zebra"] };

  const showNode = (o) => [
    S("", "if n == null", [S("", 'print "_"'), S("", "return")]), S("", "if depth > 64", [S("", 'print "..."'), S("", "return")]),
    S("", "print n.data"), ...(avl(o) ? [S("", 'print ":"'), S("", "print n.height")] : []), S("", "if n.left == null and n.right == null", [S("", "return")]),
    S("", 'print "("'), S("", "show_node(n.left, depth + 1)"), S("", 'print ", "'), S("", "show_node(n.right, depth + 1)"), S("", 'print ")"')];
  const showFn = () => [S("", "show_node(B.root, 0)"), S("", 'print " size="'), S("", "print B.size"),
    S("", "if not is_bst(B)", [S("", 'print " BROKEN: is_bst(B) is false"'), S("", "println"), S("", "return")]), S("", "println")];
  const shapeFn = () => [S("", "show_node(B.root, 0)"), S("", "println")];

  // broken trees for is_bst, from the shape the model built. The rightmost node of the left subtree (no right child)
  // made too big breaks only the upper bound; the leftmost node of the right subtree made too small, only the lower one.
  function breaks(o, m) {
    const path = (start, dir) => { let n = m.st.root[start === "left" ? "l" : "r"], p = `B.root.${start}`; while (n && n[dir === "left" ? "l" : "r"]) { n = n[dir === "left" ? "l" : "r"]; p += `.${dir}`; } return n ? p : null; };
    const hiPath = path("left", "right"), loPath = path("right", "left");
    return [[["B.size += 1"], "B.size is one too big"],
      ...(hiPath ? [[[`${hiPath}.data = ${C.lit(BIG[o.type])}`], "a node in the left subtree is bigger than the root"]] : []),
      ...(loPath ? [[[`${loPath}.data = ${C.lit(SMALL[o.type])}`], "a node in the right subtree is smaller than the root"]] : []),
      ...(avl(o) ? [[["B.root.height += 1"], "the root's height is wrong"]] : [])];
  }

  function handlers(o) {
    const nm = (n) => fname(n, o.lang);
    const { lit, say } = C;
    const shown = (st, call, label) => ({ ir: [say(`${label}: `), S("", call), S("", "show(B)")], out: `${label}: ${st.m.show()}` });
    return {
      node: (x) => ({
        ir: [S("", `let n = new(${lit(x)})`), say(`Node(${x}): data=`), S("", "print n.data"), say(", left is null: "), S("", "printb n.left == null"), say(", right is null: "), S("", "printb n.right == null"),
          ...(avl(o) ? [say(", height="), S("", "print n.height")] : []), S("", "println")],
        out: `Node(${x}): data=${x}, left is null: true, right is null: true${avl(o) ? ", height=1" : ""}` }),
      new: (x, st) => { st.m = model(o); return { ir: [S("", "make BST B"), say("new tree: "), S("", "show(B)")], out: `new tree: ${st.m.show()}` }; },
      fill: (xs, st) => { xs.forEach((y) => st.m.insert(y)); return { ir: xs.map((y) => S("", `ref_bst_insert(B, ${lit(y)})`)) }; },
      size: (x, st) => C.ops.value(o, "bst_size(B)", `${nm("bst_size")}(B)`, st.m.st.size),
      is_bst: (x, st) => C.ops.bool(o, "is_bst(B)", `${nm("is_bst")}(B)`, st.m.ok()),
      rotate: (dir, st) => {
        st.m.rotateRoot(dir);
        return { ir: [S("", `B.root = rotate_${dir}(B.root)`), say(`B.root = ${nm(`rotate_${dir}`)}(B.root): `), S("", "shape(B)")], out: `B.root = ${nm(`rotate_${dir}`)}(B.root): ${st.m.shape()}` };
      },
      lookup: (x, st) => {
        const r = st.m.lookup(x);
        if (!splay(o)) return C.ops.bool(o, `bst_lookup(B, ${lit(x)})`, `${nm("bst_lookup")}(B, ${x})`, r);
        return { ir: [say(`${nm("bst_lookup")}(B, ${x}) = `), S("", `printb bst_lookup(B, ${lit(x)})`), say(": "), S("", "show(B)")], out: `${nm("bst_lookup")}(B, ${x}) = ${r}: ${st.m.show()}` };
      },
      insert: (x, st) => { st.m.insert(x); return shown(st, `bst_insert(B, ${lit(x)})`, `${nm("bst_insert")}(B, ${x})`); },
      min: (x, st) => C.ops.value(o, "bst_min(B)", `${nm("bst_min")}(B)`, st.m.min()),
      print: (x, st) => ({ ir: [say(`${nm("bst_print")}(B) prints: `), S("", "bst_print(B)")], out: `${nm("bst_print")}(B) prints: [${st.m.inorder().join(", ")}]` }),
      delete: (x, st) => {
        const r = st.m.remove(x);
        return { ir: [say(`${nm("bst_delete")}(B, ${x}) = `), S("", `printb bst_delete(B, ${lit(x)})`), say(": "), S("", "show(B)")], out: `${nm("bst_delete")}(B, ${x}) = ${r}: ${st.m.show()}` };
      },
      break: (k, st) => {
        const B = breaks(o, st.m);
        return { ir: [...B[k][0].map((x) => S("", x)), say(`after ${B[k][1]}: ${nm("is_bst")}(B) = `), S("", "printb is_bst(B)"), S("", "println")],
          out: `after ${B[k][1]}: ${nm("is_bst")}(B) = false` };
      },
      build: (kind, st) => {
        const [a, b, c] = [...VALS[o.type].slice(0, 3)].sort((p, q) => (p < q ? -1 : 1));
        const spec = { LL: [c, 3, [b, 2, [a, 1]]], RR: [a, 3, null, [b, 2, null, [c, 1]]], LR: [c, 3, [a, 2, null, [b, 1]]], RL: [a, 3, null, [c, 2, [b, 1]]], OK: [b, 2, [a, 1], [c, 1]] }[kind];
        st.m.build(spec);
        const ir = [];
        const walk = (q, path) => {
          ir.push(S("", `${path} = new(${lit(q[0])})`));
          if (q[1] > 1) ir.push(S("", `${path}.height = ${q[1]}`));
          if (q[2]) walk(q[2], `${path}.left`);
          if (q[3]) walk(q[3], `${path}.right`);
        };
        walk(spec, "B.root");
        return { ir: [...ir, S("", "B.size = 3"), say(`${kind} shape: `), S("", "shape(B)")], out: `${kind} shape: ${st.m.shape()}` };
      },
      center: (x, st) => { st.m.splayRoot(x); return { ir: [S("", `B.root = ref_splay(B.root, ${lit(x)})`)] }; },
      rebalance_root: (x, st) => { st.m.rebalanceRoot(); return { ir: [S("", "B.root = rebalance(B.root)"), say(`B.root = ${nm("rebalance")}(B.root): `), S("", "show(B)")], out: `B.root = ${nm("rebalance")}(B.root): ${st.m.show()}` }; },
      splay_root: (x, st) => { st.m.splayRoot(x); return { ir: [S("", `B.root = splay(B.root, ${lit(x)})`), say(`B.root = splay(B.root, ${x}): `), S("", "show(B)")], out: `B.root = splay(B.root, ${x}): ${st.m.show()}` }; },
    };
  }

  function testGroups(names, o, seed) {
    const v = VALS[o.type], z = ABSENT[o.type];
    const sorted = [...v.slice(0, 7)].sort((a, b) => (a < b ? -1 : 1));
    const G = {
      tree: [[["node", v[0]]], [["new"], ["size"]]],
      // (a filled splay tree has its last insert at the root: splay the median up first, so both sides exist)
      contracts: [[["new"], ["is_bst"]], [["new"], ["fill", v.slice(0, 7)], ["is_bst"], ["size"]],
        ...breaks(o, (() => { const m = model(o); v.slice(0, 7).forEach((x) => m.insert(x)); if (splay(o)) m.splayRoot(sorted[3]); return m; })())
          .map((_, k) => [["new"], ["fill", v.slice(0, 7)], ...(splay(o) ? [["center", sorted[3]]] : []), ["break", k]]),
        // AVL: correct heights but out of balance (a right-right chain)
        ...(avl(o) ? [[["new"], ["build", "RR"], ["is_bst"]]] : [])],
      rotations: [(() => {   // rotations the tree's shape allows (a filled splay tree has its last insert at the root)
        const m = model(o);
        v.slice(0, 7).forEach((x) => m.insert(x));
        const ops = [["new"], ["fill", v.slice(0, 7)]];
        for (const dir of ["right", "left", "left", "right", "right"]) {
          if (!(dir === "right" ? m.st.root.l : m.st.root.r)) continue;
          m.rotateRoot(dir);
          ops.push(["rotate", dir]);
        }
        return ops;
      })()],
      // the four unbalanced shapes (and a balanced one), built by hand, then rebalanced at the root
      rebalance: ["LL", "RR", "LR", "RL", "OK"].map((kind) => [["new"], ["build", kind], ["rebalance_root"]]),
      splay: [[["new"], ["fill", v.slice(0, 7)], ["splay_root", sorted[0]], ["splay_root", sorted[6]], ["splay_root", sorted[3]], ["splay_root", z[1]],
        ["splay_root", z[0]], ["splay_root", z[2]]]],
      lookup: [[["new"], ["lookup", v[0]], ["fill", v.slice(0, 7)], ...v.slice(0, 7).map((x) => ["lookup", x]), ...z.map((x) => ["lookup", x])]],
      insert: [[["new"], ...v.slice(0, 9).map((x) => ["insert", x]), ["insert", v[2]], ["size"]], [["new"], ...sorted.slice(0, 5).map((x) => ["insert", x])]],
      min_print: [[["new"], ["print"], ["fill", v.slice(0, 9)], ["min"], ["print"], ["insert", z[0]], ["min"], ["print"]]],
      delete: [
        (() => {
          const m = model(o), fill = v.slice(0, 10);
          fill.forEach((x) => m.insert(x));
          const ops = [["new"], ["fill", fill], ["delete", z[1]]];
          const kinds = [(n) => !n.l && !n.r, (n) => n.l && !n.r, (n) => !n.l && n.r, (n) => n.l && n.r, (n) => n === m.st.root];
          let last = null;
          for (const kind of kinds) {
            const hit = m.nodes().find(([n]) => kind(n));
            if (!hit) continue;
            last = hit[0].d;
            ops.push(["delete", last]);
            m.remove(last);
          }
          return [...ops, ["delete", last], ["size"], ["insert", v[0]]];
        })(),
        [["new"], ["fill", sorted.slice(0, 4)], ...sorted.slice(0, 4).map((x) => ["delete", x]), ["delete", sorted[0]], ["insert", sorted[5]]],
        [["new"], ["fill", [sorted[1], sorted[3], sorted[2]]], ["delete", sorted[1]]],
        [["new"], ["fill", [sorted[3], sorted[1], sorted[2]]], ["delete", sorted[3]]],
      ],
    };
    const groups = names.flatMap((n) => G[n] || []);
    if (seed) {
      const r = C.mulberry32(seed);
      const pool = [...v, ...z];
      const ops = [["new"]];
      for (let i = 0; i < 50; i++) {
        const x = r(), k = pool[Math.floor(r() * pool.length)];
        ops.push(x < 0.5 ? ["insert", k] : x < 0.75 ? ["delete", k] : ["lookup", k]);
      }
      ops.push(["print"]);
      groups.push(ops);
    }
    return groups;
  }

  // the checker's own helpers, built on the user's Node and BST
  function refFns(o, names) {
    if (!names.includes("tree")) return [];
    const A = avl(o), SP = splay(o);
    return [
      { name: "is_ordered", steps: isOrdered() }, { name: "node_size", steps: nodeSize() },
      ...(A ? [{ name: "height", steps: height() }, { name: "fix_height", steps: fixHeight() }, { name: "is_avl", steps: isAvl() }] : []),
      { name: "is_bst", steps: isBst(o)() },
      ...(A || SP ? [{ name: "rotate_right", steps: rot(o, "right")() }, { name: "rotate_left", steps: rot(o, "left")() }] : []),
      ...(A ? [{ name: "rebalance", steps: rebalance() }] : []),
      ...(SP ? [{ name: "splay", steps: splayFn() }] : [{ name: "node_lookup", steps: nodeLookup() }, { name: "node_insert", steps: nodeInsert(o)() }]),
      { name: "bst_lookup", steps: lookup(o)() }, { name: "bst_insert", steps: insert(o)() },
      { name: "show_node", steps: showNode(o) }, { name: "show", steps: showFn() }, { name: "shape", steps: shapeFn() },
    ];
  }

  function problem(o) {
    const L = o.lang, A = avl(o), SP = splay(o);
    const sig = C.headers(DEF, o);
    const what = {
      bst_size: "the number of elements",
      ...(A ? { height: "a subtree's height (0 when empty)", fix_height: "1 + the taller child's height" } : {}),
      is_ordered: "every node strictly between the bounds lo and hi (nodes, or null for none)", node_size: "the number of nodes",
      ...(A ? { is_avl: "heights correct and children's heights within 1" } : {}), is_bst: "the representation invariant",
      ...(A || SP ? { rotate_right: "lift the left child; return the new subtree root", rotate_left: "lift the right child; return the new subtree root" } : {}),
      ...(A ? { rebalance: "fix the height; rotate once (outside case) or twice (inside case) when one side is 2 taller" } : {}),
      ...(SP ? { splay: "bring x (or the last node on its path) to the subtree's root with zig, zig-zig and zig-zag" } : { node_lookup: "recursive search" }),
      bst_lookup: SP ? "splay x to the root and check it" : "true if x is in the tree",
      ...(SP ? {} : { node_insert: "insert below n; return the subtree's root" + (A ? ", rebalanced" : "") }),
      bst_insert: SP ? "splay, then split the tree around a new root (nothing if x is there)" : "insert x if it isn't there",
      node_min: "the leftmost node", bst_min: "the smallest element (requires a non-empty tree)",
      node_print: "print the subtree in order after k earlier elements; return the new count", bst_print: "print the elements sorted, like `[3, 5, 7]`, and end the line",
      ...(SP ? {} : { node_delete_min: "remove the leftmost node; return the subtree's root", node_delete: "remove x's node (two children: copy in the right subtree's minimum)" }),
      bst_delete: SP ? "splay x to the root, remove it, join the subtrees; false if x isn't there" : "remove x; false if it isn't there",
    };
    return [
      `Build a **${describe(o)}** from a blank file, with no hints: everything the tutorial walked you through.`, "",
      `**Node.** Fields \`data\`, \`left\`, \`right\`${A ? ", `height` (1 for a new node)" : ""}. **BST.** Fields \`root\` and \`size\`; ` +
        `${{ py: "`BST()`", cpp: "`BST B;`", java: "`new BST()`", rs: "`BST::new()`" }[L]} makes an empty tree.`, "",
      "**Functions** (exact names and signatures):", "",
      ...Object.keys(what).map((n) => `* \`${sig(n)}\`: ${what[n]}`), "",
      `Check \`${fname("is_bst", L)}(B)\` at the start and end of every operation that changes the tree. ${L === "java" ? "Put the functions inside `class BST` as static methods. " : ""}` +
        "You don't need a `main`: the checker adds its own (yours, if any, is ignored), prints the tree's shape after every change, and runs every test from the tutorial plus random ones.",
    ].join("\n");
  }

  const DEF = {
    id: "binary-search-trees", num: "13.09", title: "Binary search trees", heading: "Build a binary search tree, step by step", diagramLabel: "Your tree",
    OPTIONS, DEFAULTS, FNS, STRUCTS, ctxExtra, describe, complexity, diagram, steps, handlers, testGroups, refFns, problem,
    CONTRACT_OPS: ["is_bst", "break"],
  };
  const tutorial = C.make(DEF);
  if (typeof module !== "undefined" && module.exports) module.exports = tutorial;
})(typeof window !== "undefined" ? window : globalThis);
