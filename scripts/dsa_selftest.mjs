// Self-test for the Data Structures tutorials (docs/dsa/): every tutorial's reference solution, in every language and
// every combination of options, must pass the checker's tests, step by step and as the final challenge; and (with
// --mutants) dropping any one statement from a Python reference must make the final tests fail.
//
//   node scripts/dsa_selftest.mjs                 all steps + final, all languages (needs clang++, javac/java, rustc)
//   node scripts/dsa_selftest.mjs --quick         final challenge only
//   node scripts/dsa_selftest.mjs --lang py,cpp   some languages
//   node scripts/dsa_selftest.mjs --mutants       mutation check (Python)
//   node scripts/dsa_selftest.mjs --only stacks   some tutorials
//
// C++, Java and Rust run locally here (the site sends them to Compiler Explorer); Python runs generator/api.py's
// dsa_run, the function the site's Pyodide worker calls.
import { execFile } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync, existsSync } from "node:fs";
import { tmpdir, cpus } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(import.meta.url);
require(join(ROOT, "docs/dsa/lang.js"));
require(join(ROOT, "docs/dsa/common.js"));
const RUN = require(join(ROOT, "docs/dsa/run.js"));
const args = process.argv.slice(2);
const flag = (f) => args.includes(f);
const opt = (f, d) => { const i = args.indexOf(f); return i >= 0 ? args[i + 1] : d; };
// every tutorial in docs/dsa/ (--only linked-lists,stacks to pick some)
const TUTORIALS = Object.fromEntries(["binary-search", "linked-list", "stack", "queue"]
  .filter((f) => existsSync(join(ROOT, `docs/dsa/${f}.js`)))
  .map((f) => { const t = require(join(ROOT, `docs/dsa/${f}.js`)); return [t.id, t]; })
  .filter(([id]) => !opt("--only", "") || opt("--only", "").split(",").includes(id)));
const PYTHON = process.env.PYTHON || join(ROOT, ".venv/bin/python");

const LANGS = opt("--lang", "py,cpp,java,rs").split(",");

function sh(cmd, argv, opts = {}) {
  return new Promise((resolve) => {
    execFile(cmd, argv, { timeout: 20000, maxBuffer: 16 << 20, ...opts }, (err, stdout, stderr) =>
      resolve({ code: err ? (err.code ?? 1) : 0, stdout, stderr, killed: !!(err && err.killed) }));
  });
}

// every combination of a tutorial's options
function combos(t) {
  let out = [{}];
  for (const o of t.OPTIONS) out = out.flatMap((c) => o.choices.map(([v]) => ({ ...c, [o.id]: v })));
  return out.filter((c) => LANGS.includes(c.lang));
}

async function runCompiled(lang, source) {
  const dir = mkdtempSync(join(tmpdir(), "dsa-"));
  try {
    if (lang === "cpp") {
      writeFileSync(join(dir, "prog.cpp"), source);
      const b = await sh("clang++", ["-std=c++20", "-O1", "-w", "-o", join(dir, "prog"), join(dir, "prog.cpp")]);
      if (b.code) return { compiled: false, compileOutput: b.stderr };
      const r = await sh(join(dir, "prog"), []);
      return { compiled: true, stdout: r.stdout, stderr: r.stderr, exit: r.code };
    }
    if (lang === "java") {
      writeFileSync(join(dir, "prog.java"), source);
      const b = await sh("javac", ["-nowarn", "-d", join(dir, "out"), join(dir, "prog.java")]);
      if (b.code) return { compiled: false, compileOutput: b.stderr };
      const r = await sh("java", ["-Xss1m", "-cp", join(dir, "out"), "TLTests"]);
      return { compiled: true, stdout: r.stdout, stderr: r.stderr, exit: r.code };
    }
    writeFileSync(join(dir, "prog.rs"), source);
    const b = await sh("rustc", ["--edition", "2021", "-A", "warnings", "-o", join(dir, "prog"), join(dir, "prog.rs")]);
    if (b.code) return { compiled: false, compileOutput: b.stderr };
    const r = await sh(join(dir, "prog"), []);
    return { compiled: true, stdout: r.stdout, stderr: r.stderr, exit: r.code };
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

// Python jobs go through one interpreter: [{user, tests}] -> [result]
async function runPythonBatch(jobs) {
  if (!jobs.length) return [];
  const dir = mkdtempSync(join(tmpdir(), "dsa-py-"));
  try {
    writeFileSync(join(dir, "jobs.json"), JSON.stringify(jobs));
    // each job gets 5 s (a mutant can loop forever); the alarm surfaces as an error in that job's result
    const script = `import json, signal, sys\nsys.path.insert(0, ${JSON.stringify(ROOT)})\nfrom generator.api import dsa_run\n` +
      `def alarm(*_): raise TimeoutError("time limit")\nsignal.signal(signal.SIGALRM, alarm)\n` +
      `def one(j):\n    signal.setitimer(signal.ITIMER_REAL, 5)\n    try: return json.loads(dsa_run(j["user"], j["tests"]))\n    finally: signal.setitimer(signal.ITIMER_REAL, 0)\n` +
      `jobs = json.load(open(${JSON.stringify(join(dir, "jobs.json"))}))\nprint(json.dumps([one(j) for j in jobs]))\n`;
    const r = await sh(PYTHON, ["-c", script], { timeout: 600000 });
    if (r.code) throw new Error(`python failed: ${r.stderr}`);
    return JSON.parse(r.stdout);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

async function pool(items, n, fn) {
  const out = new Array(items.length);
  let i = 0;
  await Promise.all(Array.from({ length: n }, async () => { while (i < items.length) { const k = i++; out[k] = await fn(items[k], k); } }));
  return out;
}

function verdict(res, expected) {
  if (!res.compiled) return `doesn't compile:\n${res.compileOutput}`;
  const cmp = RUN.compare(expected, res.stdout);
  if (cmp.ok && !res.stderr?.trim()) return null;
  if (cmp.ok) return `passed but wrote to stderr:\n${res.stderr}`;
  return `line ${cmp.line + 1}: expected ${JSON.stringify(cmp.expected)}, got ${JSON.stringify(cmp.got)}\n${(res.stderr || "").slice(0, 2000)}`;
}

async function selftest() {
  const jobs = [];
  for (const [name, t] of Object.entries(TUTORIALS)) {
    for (const o of combos(t)) {
      const ups = flag("--quick") ? [] : Array.from({ length: t.stepCount(o) }, (_, k) => k + 1);
      for (const upto of ups) jobs.push({ name, o, upto, final: false });
      jobs.push({ name, o, upto: t.stepCount(o), final: true, seed: 7 });
    }
  }
  const build = (j) => {
    const t = TUTORIALS[j.name];
    const h = t.harness(j.o, j.upto, { final: j.final, seed: j.seed });
    const user = t.program(j.o, j.upto);
    return { h, prog: RUN.assemble(j.o.lang, user, h) };
  };
  const py = jobs.filter((j) => j.o.lang === "py");
  const other = jobs.filter((j) => j.o.lang !== "py");
  const failures = [];
  const pyRes = await runPythonBatch(py.map((j) => build(j).prog));
  py.forEach((j, k) => { const v = verdict(pyRes[k], build(j).h.expected); if (v) failures.push([j, v]); });
  let done = 0;
  await pool(other, Math.max(2, cpus().length - 2), async (j) => {
    const { h, prog } = build(j);
    const v = verdict(await runCompiled(j.o.lang, prog.source), h.expected);
    if (v) failures.push([j, v]);
    if (++done % 50 === 0) process.stderr.write(`  ${done}/${other.length} compiled runs\n`);
  });
  for (const [j, v] of failures.slice(0, 12)) console.log(`FAIL ${j.name} ${JSON.stringify(j.o)} ${j.final ? "final" : `step ${j.upto}`}: ${v}\n`);
  console.log(`${jobs.length} runs, ${failures.length} failures`);
  return failures.length === 0;
}

// Drop each statement of each function in turn: the final tests must notice.
async function mutants() {
  const jobs = [];
  for (const [name, t] of Object.entries(TUTORIALS)) {
    for (const o of combos(t).filter((c) => c.lang === "py")) {
      const base = t.steps(o);
      const paths = [];
      const walk = (list, path) => list.forEach((st, i) => {
        const p = [...path, i];
        // (a block's only statement goes with the block itself)
        if (st.ir && !/^(assert|free|let x = take|else)\b/.test(st.ir) && (list.length > 1 || !path.length)) paths.push(p);
        if (st.body) walk(st.body, [...p, "body"]);
      });
      base.forEach((st, si) => (st.fns || []).forEach((f, fi) => walk(f.steps, [si, "fns", fi, "steps"])));
      for (const p of paths) {
        // a deep copy that keeps functions (decl, and step text that depends on the language)
        const clone = (x) => (Array.isArray(x) ? x.map(clone) : x && typeof x === "object" ? Object.fromEntries(Object.entries(x).map(([k, v]) => [k, clone(v)])) : x);
        const list = clone(base);
        let parent = list;
        for (const k of p.slice(0, -1)) parent = parent[k];
        const removed = parent.splice(p[p.length - 1], 1)[0];
        let user;
        try { user = t.program(o, t.stepCount(o), list); } catch (e) { continue; }
        const h = t.harness(o, t.stepCount(o), { final: true, seed: 7 });
        jobs.push({ o, p, removed: removed.ir, prog: RUN.assemble("py", user, h), expected: h.expected, fn: base[p[0]].fns[p[2]].name });
      }
    }
  }
  const res = await runPythonBatch(jobs.map((j) => j.prog));
  const survivors = jobs.filter((j, k) => res[k].compiled && RUN.compare(j.expected, res[k].stdout).ok && !res[k].stderr);
  for (const s of survivors) console.log(`SURVIVED ${JSON.stringify(s.o)} ${s.fn}: dropping "${s.removed}"`);
  console.log(`${jobs.length} mutants, ${survivors.length} survived`);
  return survivors.length === 0;
}

const ok = flag("--mutants") ? await mutants() : await selftest();
process.exit(ok ? 0 : 1);
