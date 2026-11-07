"""Databases for the SQL problems: the guide's shop (users, orders), plus ML-flavoured ones (experiment runs and
metrics, a GPU job queue, a model registry with evals, product events with JSONB payloads, classifier predictions).

Every builder takes an rng and a set of traps and returns (setup SQL, facts). The shown database is small enough to
read; hidden test databases come from the same builder with other seeds and traps that catch the classic mistakes:
users without orders, NULLs, ties, values exactly on a boundary, duplicate names, refunded-only users, empty groups.
"""

from __future__ import annotations

import datetime as dt
import json
import random
from decimal import Decimal

NAMES = ["Ana", "Ben", "Cal", "Dee", "Eve", "Fay", "Gus", "Hal", "Ivy", "Jon", "Kim", "Lea", "Max", "Nia", "Oli",
         "Pia", "Quinn", "Ray", "Sam", "Tia", "Uma", "Vic", "Wes", "Xia", "Yuri", "Zoe", "Abe", "Bea", "Cy", "Dot"]
COUNTRIES = ["US", "UK", "DE", "FR", "IN", "JP", "BR", "CA"]
STATUSES = ["paid", "paid", "paid", "paid", "refunded", "pending"]


def lit(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, Decimal)):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, dt.datetime):
        return "'" + v.strftime("%Y-%m-%d %H:%M:%S") + ("" if v.tzinfo is None else "+00") + "'"
    if isinstance(v, dt.date):
        return "'" + v.isoformat() + "'"
    if isinstance(v, (dict, list)) and not isinstance(v, Arr):
        return "'" + json.dumps(v).replace("'", "''") + "'"
    if isinstance(v, Arr):
        return "ARRAY[" + ", ".join(lit(x) for x in v) + "]" + (f"::{v.typ}[]" if not v else "")
    return "'" + str(v).replace("'", "''") + "'"


class Arr(list):
    """A SQL array literal (ARRAY[...]); typ names the element type for an empty one."""

    def __init__(self, items, typ="text"):
        super().__init__(items)
        self.typ = typ


def insert(table, cols, rows, per_line=4) -> str:
    if not rows:
        return ""
    vals = ["(" + ", ".join(lit(v) for v in r) + ")" for r in rows]
    lines = []
    for i in range(0, len(vals), per_line):
        lines.append("  " + ", ".join(vals[i:i + per_line]))
    return f"INSERT INTO {table} ({', '.join(cols)}) VALUES\n" + ",\n".join(lines) + ";\n"


def money(rng, lo=5, hi=300):
    return Decimal(rng.randint(lo * 100, hi * 100)) / 100


# ---------------------------------------------------------------- shop (COMP_GUIDE 10.00)

SHOP_DDL = """CREATE TABLE users (
  id          INTEGER PRIMARY KEY,
  name        TEXT NOT NULL,
  country     TEXT,
  signup_date DATE NOT NULL
);
CREATE TABLE orders (
  id         INTEGER PRIMARY KEY,
  user_id    INTEGER NOT NULL REFERENCES users(id),
  amount     NUMERIC(10, 2) NOT NULL CHECK (amount >= 0),
  status     TEXT NOT NULL DEFAULT 'paid',
  created_at TIMESTAMP NOT NULL
);
"""


def shop(rng, n_users=7, n_orders=14, traps=(), start=dt.date(2024, 1, 1), days=120, like=None):
    """users + orders. Always: one user without orders, one NULL country (unless the traps say otherwise)."""
    traps = set(traps)
    names = rng.sample(NAMES, n_users) if n_users <= len(NAMES) else [f"user{i}" for i in range(n_users)]
    if "dup_names" in traps and n_users > 2:
        names[-1] = names[0]
    countries = (like or {}).get("countries") or rng.sample(COUNTRIES, 4)
    users = []
    for i, nm in enumerate(names, 1):
        c = rng.choice(countries)
        users.append([i, nm, c, start + dt.timedelta(days=rng.randrange(days // 2))])
    nulls = 1 + ("null_country" in traps) * 2
    for k in rng.sample(range(n_users), min(nulls, n_users - 1)):
        users[k][2] = None
    if "lower_case" in traps:
        users[rng.randrange(n_users)][1] = names[0].lower()
    no_orders = {users[-1][0]} if "all_ordered" not in traps else set()
    if "no_orders" in traps and n_users > 3:
        no_orders.add(users[rng.randrange(n_users - 1)][0])
    buyers = [u for u in users if u[0] not in no_orders]
    orders = []
    t0 = dt.datetime(start.year, start.month, start.day)
    for i in range(1, n_orders + 1):
        u = rng.choice(buyers)
        signup = dt.datetime(u[3].year, u[3].month, u[3].day)
        when = signup + dt.timedelta(minutes=rng.randrange(60, 60 * 24 * (days // 2)))
        when = when.replace(minute=(when.minute // 5) * 5, second=0)
        orders.append([i, u[0], money(rng), rng.choice(STATUSES), when])
    if "ties" in traps and len(orders) > 3:
        a, b = rng.sample(range(len(orders)), 2)
        orders[b][2] = orders[a][2]
        c, d = rng.sample(range(len(orders)), 2)
        orders[d][4] = orders[c][4]
    if "boundary" in traps:
        for k in rng.sample(range(len(orders)), min(3, len(orders))):
            w = orders[k][4]
            orders[k][4] = dt.datetime(w.year, w.month, 1)
    if "refunded_only" in traps and len(buyers) > 2:
        victim = buyers[rng.randrange(len(buyers))][0]
        for o in orders:
            if o[1] == victim:
                o[3] = "refunded"
    if "zero" in traps:
        orders[rng.randrange(len(orders))][2] = Decimal("0.00")
    orders.sort(key=lambda o: o[0])
    sql = SHOP_DDL + insert("users", ["id", "name", "country", "signup_date"], users) + \
        insert("orders", ["id", "user_id", "amount", "status", "created_at"], orders, per_line=3)
    facts = {"users": users, "orders": orders, "countries": countries, "no_orders": sorted(no_orders),
             "start": start}
    return sql, facts


SHOP_TRAPS = [("users without orders, NULL countries", ("no_orders", "null_country")),
              ("ties and orders exactly on month boundaries", ("ties", "boundary")),
              ("a user whose orders were all refunded, duplicate names", ("refunded_only", "dup_names")),
              ("more users and orders", ("ties", "no_orders", "zero"))]


def hidden(builder, rng, facts, specs=SHOP_TRAPS, k=3, **kw):
    """Hidden test databases: [(setup SQL, tag)]."""
    out = []
    for tag, traps in specs[:k]:
        seed = rng.randrange(1 << 30)
        r = random.Random(seed)
        size = kw.copy()
        if "more" in tag:
            for key, v in list(size.items()):
                if isinstance(v, int) and key.startswith("n"):
                    size[key] = v * 3
        sql, _ = builder(r, traps=traps, like=facts, **size)
        out.append((sql, tag))
    return out


# ---------------------------------------------------------------- experiment runs + metrics

MODELS = ["resnet18", "resnet50", "vit-b16", "bert-base", "gpt2-small", "t5-small", "llama-7b", "mistral-7b",
          "distilbert", "whisper-tiny"]
DATASETS = ["cifar10", "imagenet", "mnist", "squad", "glue", "coco", "wikitext"]
OPTIMS = ["sgd", "adam", "adamw"]
OWNERS = ["ada", "alan", "grace", "linus", "yann", "fei", "andrej", "ilya"]
PROJECTS = ["vision", "nlp", "speech"]

RUNS_DDL = """CREATE TABLE runs (
  run_id      INTEGER PRIMARY KEY,
  project     TEXT NOT NULL,
  model       TEXT NOT NULL,
  dataset     TEXT NOT NULL,
  optimizer   TEXT NOT NULL,
  lr          NUMERIC(8, 6) NOT NULL,
  batch_size  INTEGER NOT NULL,
  epochs      INTEGER NOT NULL,
  status      TEXT NOT NULL,
  owner       TEXT NOT NULL,
  started_at  TIMESTAMP NOT NULL,
  finished_at TIMESTAMP,
  config      JSONB,
  tags        TEXT[]
);
CREATE TABLE metrics (
  run_id   INTEGER NOT NULL REFERENCES runs(run_id),
  step     INTEGER NOT NULL,
  split    TEXT NOT NULL CHECK (split IN ('train', 'val')),
  loss     NUMERIC(8, 4) NOT NULL,
  accuracy NUMERIC(5, 4),
  PRIMARY KEY (run_id, step, split)
);
"""

TAGS = ["baseline", "sweep", "ablation", "prod", "debug", "fp16", "long"]


def runs(rng, n_runs=8, steps=3, traps=(), like=None, start=dt.datetime(2024, 3, 1, 9, 0)):
    traps = set(traps)
    models = (like or {}).get("models") or rng.sample(MODELS, 4)
    data_sets = (like or {}).get("datasets") or rng.sample(DATASETS, 3)
    rows, mrows = [], []
    for i in range(1, n_runs + 1):
        model = rng.choice(models)
        ds = rng.choice(data_sets)
        opt = rng.choice(OPTIMS)
        lr = rng.choice([Decimal("0.1"), Decimal("0.01"), Decimal("0.001"), Decimal("0.0003"), Decimal("0.0001")])
        bs = rng.choice([16, 32, 64, 128])
        ep = rng.choice([5, 10, 20])
        status = rng.choice(["finished", "finished", "finished", "failed", "running"])
        owner = rng.choice(OWNERS[:5])
        s = start + dt.timedelta(hours=rng.randrange(0, 24 * 20), minutes=rng.choice([0, 15, 30, 45]))
        fin = None if status == "running" else s + dt.timedelta(minutes=rng.randrange(20, 600))
        cfg = {"lr": float(lr), "batch_size": bs, "optimizer": opt, "dropout": rng.choice([0.0, 0.1, 0.2]),
               "warmup": rng.choice([0, 100, 500])}
        if rng.random() < 0.3:
            cfg["scheduler"] = rng.choice(["cosine", "linear"])
        tags = Arr(sorted(rng.sample(TAGS, rng.randint(0, 3))))
        rows.append([i, rng.choice(PROJECTS), model, ds, opt, lr, bs, ep, status, owner, s, fin, cfg, tags])
        if status == "failed" and "failed_no_metrics" in traps:
            continue
        loss = rng.uniform(1.2, 2.5)
        for k in range(1, steps + 1):
            loss *= rng.uniform(0.7, 0.95)
            val = loss * rng.uniform(1.0, 1.2)
            acc = None if (status == "running" and k == steps) else Decimal(str(round(rng.uniform(0.5, 0.95), 4)))
            mrows.append([i, k * 100, "train", Decimal(str(round(loss, 4))), None])
            mrows.append([i, k * 100, "val", Decimal(str(round(val, 4))), acc])
    if "ties" in traps and len(mrows) > 4:
        vals = [m for m in mrows if m[2] == "val"]
        a, b = rng.sample(vals, 2)
        b[3] = a[3]
    sql = RUNS_DDL + insert("runs", ["run_id", "project", "model", "dataset", "optimizer", "lr", "batch_size",
                                     "epochs", "status", "owner", "started_at", "finished_at", "config", "tags"],
                            rows, per_line=1) + \
        insert("metrics", ["run_id", "step", "split", "loss", "accuracy"], mrows, per_line=4)
    return sql, {"runs": rows, "metrics": mrows, "models": models, "datasets": data_sets}


RUNS_TRAPS = [("failed runs without metrics, running runs", ("failed_no_metrics",)),
              ("ties in validation loss", ("ties",)),
              ("more runs", ("ties", "failed_no_metrics"))]


# ---------------------------------------------------------------- GPU jobs

GPU_DDL = """CREATE TABLE gpu_jobs (
  job_id       INTEGER PRIMARY KEY,
  user_name    TEXT NOT NULL,
  team         TEXT NOT NULL,
  gpu_type     TEXT NOT NULL,
  gpu_count    INTEGER NOT NULL CHECK (gpu_count > 0),
  submitted_at TIMESTAMP NOT NULL,
  started_at   TIMESTAMP,
  ended_at     TIMESTAMP,
  state        TEXT NOT NULL
);
"""
TEAMS = ["vision", "llm", "speech", "infra"]
GPUS = ["A100", "H100", "L4", "T4"]


def gpu_jobs(rng, n_jobs=14, traps=(), like=None, start=dt.datetime(2024, 5, 6, 8, 0), span_hours=24 * 7):
    traps = set(traps)
    users = (like or {}).get("users") or rng.sample(OWNERS, 5)
    team_of = {u: rng.choice(TEAMS) for u in users}
    rows = []
    for i in range(1, n_jobs + 1):
        u = rng.choice(users)
        sub = start + dt.timedelta(minutes=rng.randrange(0, 60 * span_hours))
        sub = sub.replace(minute=sub.minute // 5 * 5, second=0)
        state = rng.choice(["done", "done", "done", "failed", "running", "queued", "preempted"])
        st = None if state == "queued" else sub + dt.timedelta(minutes=rng.randrange(1, 240))
        en = None if state in ("queued", "running") else st + dt.timedelta(minutes=rng.randrange(10, 600))
        rows.append([i, u, team_of[u], rng.choice(GPUS), rng.choice([1, 1, 2, 4, 8]), sub, st, en, state])
    if "ties" in traps:
        a, b = rng.sample(range(len(rows)), 2)
        rows[b][4] = rows[a][4]
        rows[b][5] = rows[a][5]
    if "idle_team" in traps:
        pass
    sql = GPU_DDL + insert("gpu_jobs", ["job_id", "user_name", "team", "gpu_type", "gpu_count", "submitted_at",
                                        "started_at", "ended_at", "state"], rows, per_line=1)
    return sql, {"jobs": rows, "users": users}


GPU_TRAPS = [("queued jobs that never started", ()), ("ties", ("ties",)), ("more jobs", ("ties",))]


# ---------------------------------------------------------------- model registry + evals

MODELS_DDL = """CREATE TABLE models (
  model_id    INTEGER PRIMARY KEY,
  name        TEXT NOT NULL UNIQUE,
  family      TEXT NOT NULL,
  params_b    NUMERIC(6, 2) NOT NULL,
  org         TEXT NOT NULL,
  released_on DATE NOT NULL,
  parent_id   INTEGER REFERENCES models(model_id)
);
CREATE TABLE evals (
  model_id     INTEGER NOT NULL REFERENCES models(model_id),
  benchmark    TEXT NOT NULL,
  score        NUMERIC(5, 2) NOT NULL,
  evaluated_at DATE NOT NULL,
  PRIMARY KEY (model_id, benchmark)
);
"""
FAMILIES = [("llama", "meta"), ("mistral", "mistral"), ("qwen", "alibaba"), ("gemma", "google"), ("phi", "microsoft")]
BENCHES = ["mmlu", "gsm8k", "humaneval", "arc"]


def model_registry(rng, n_base=4, n_ft=5, traps=(), like=None):
    traps = set(traps)
    fams = rng.sample(FAMILIES, 3)
    rows = []
    mid = 0
    base_ids = []
    for k in range(n_base):
        fam, org = fams[k % len(fams)]
        size = rng.choice([Decimal("1.5"), Decimal("3"), Decimal("7"), Decimal("8"), Decimal("13"), Decimal("70")])
        mid += 1
        name = f"{fam}-{size.normalize()}b"
        if any(r[1] == name for r in rows):
            name += f"-v{mid}"
        rows.append([mid, name, fam, size, org, dt.date(2023, 1, 1) + dt.timedelta(days=rng.randrange(500)), None])
        base_ids.append(mid)
    for k in range(n_ft):
        parent = rng.choice([r for r in rows])
        mid += 1
        suffix = rng.choice(["instruct", "chat", "code", "math", "dpo"])
        name = f"{parent[1]}-{suffix}"
        if any(r[1] == name for r in rows):
            name += f"-{mid}"
        rows.append([mid, name, parent[2], parent[3], parent[4], parent[5] + dt.timedelta(days=rng.randrange(10, 200)),
                     parent[0]])
    evals = []
    for r in rows:
        for b in BENCHES:
            if rng.random() < 0.8 or b == "mmlu":
                evals.append([r[0], b, Decimal(rng.randint(2000, 9000)) / 100,
                              r[5] + dt.timedelta(days=rng.randrange(1, 30))])
    if "ties" in traps:
        mm = [e for e in evals if e[1] == "mmlu"]
        if len(mm) > 2:
            mm[1][2] = mm[0][2]
    if "unevaluated" in traps:
        evals = [e for e in evals if e[0] != rows[-1][0]]
    sql = MODELS_DDL + insert("models", ["model_id", "name", "family", "params_b", "org", "released_on", "parent_id"],
                              rows, per_line=1) + \
        insert("evals", ["model_id", "benchmark", "score", "evaluated_at"], evals, per_line=3)
    return sql, {"models": rows, "evals": evals}


MODEL_TRAPS = [("a model without evals", ("unevaluated",)), ("tied scores", ("ties",)),
               ("deeper lineage", ("ties", "unevaluated"))]


# ---------------------------------------------------------------- product events (JSONB)

EVENTS_DDL = """CREATE TABLE events (
  event_id   INTEGER PRIMARY KEY,
  user_id    INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  ts         TIMESTAMP NOT NULL,
  payload    JSONB NOT NULL
);
"""
PAGES = ["/home", "/pricing", "/docs", "/signup", "/checkout", "/blog"]
DEVICES = ["mobile", "desktop", "tablet"]


def events(rng, n_users=5, per_user=5, traps=(), like=None, start=dt.datetime(2024, 4, 1, 9, 0)):
    traps = set(traps)
    rows = []
    eid = 0
    for u in range(1, n_users + 1):
        t = start + dt.timedelta(days=rng.randrange(5), minutes=rng.randrange(0, 600))
        n = per_user + rng.randint(-2, 2)
        for k in range(max(1, n)):
            eid += 1
            et = "signup" if k == 0 else rng.choice(["view", "view", "click", "click", "purchase"])
            gap = rng.choice([1, 2, 5, 10, 40, 90, 600])
            t = t + dt.timedelta(minutes=gap)
            payload = {"page": rng.choice(PAGES), "device": rng.choice(DEVICES)}
            if et == "purchase":
                payload["amount"] = round(rng.uniform(5, 200), 2)
                payload["items"] = rng.sample(["gpu", "cable", "ssd", "fan", "psu"], rng.randint(1, 3))
            if et == "click" and rng.random() < 0.5:
                payload["button"] = rng.choice(["buy", "try", "docs"])
            if "missing_key" in traps and rng.random() < 0.2:
                payload.pop("device", None)
            rows.append([eid, u, et, t.replace(second=0), payload])
    sql = EVENTS_DDL + insert("events", ["event_id", "user_id", "event_type", "ts", "payload"], rows, per_line=1)
    return sql, {"events": rows}


EVENT_TRAPS = [("payloads missing a key", ("missing_key",)), ("other sessions", ()), ("more users", ())]


# ---------------------------------------------------------------- predictions

PRED_DDL = """CREATE TABLE predictions (
  item_id       INTEGER PRIMARY KEY,
  model_version TEXT NOT NULL,
  label         BOOLEAN NOT NULL,
  score         NUMERIC(4, 3) NOT NULL CHECK (score BETWEEN 0 AND 1),
  predicted_at  TIMESTAMP NOT NULL
);
"""


def predictions(rng, n=16, traps=(), like=None, start=dt.datetime(2024, 6, 1)):
    traps = set(traps)
    versions = (like or {}).get("versions") or ["v1", "v2"]
    rows = []
    for i in range(1, n + 1):
        label = rng.random() < 0.4
        base = 0.65 if label else 0.35
        score = min(1.0, max(0.0, rng.gauss(base, 0.2)))
        rows.append([i, rng.choice(versions), label, Decimal(str(round(score, 3))),
                     start + dt.timedelta(hours=rng.randrange(0, 240))])
    if "boundary" in traps:
        rows[0][3] = Decimal("0.500")
        rows[1][3] = Decimal("1.000")
    if "ties" in traps:
        rows[3][3] = rows[2][3]
    sql = PRED_DDL + insert("predictions", ["item_id", "model_version", "label", "score", "predicted_at"], rows,
                            per_line=2)
    return sql, {"rows": rows, "versions": versions}


PRED_TRAPS = [("scores exactly on the threshold", ("boundary",)), ("tied scores", ("ties",)),
              ("more predictions", ("boundary", "ties"))]


# ---------------------------------------------------------------- ad campaigns (safe division, ratios)

CAMPAIGNS_DDL = """CREATE TABLE campaigns (
  id          INTEGER PRIMARY KEY,
  name        TEXT NOT NULL,
  channel     TEXT NOT NULL,
  impressions INTEGER NOT NULL,
  clicks      INTEGER NOT NULL,
  spend       NUMERIC(10, 2) NOT NULL,
  conversions INTEGER
);
"""
CHANNELS = ["search", "social", "video", "email"]
ADJ = ["spring", "launch", "retarget", "brand", "promo", "holiday", "beta", "webinar"]


def campaigns(rng, n=8, traps=(), like=None):
    traps = set(traps)
    rows = []
    names = rng.sample(ADJ, min(n, len(ADJ))) + [f"test{k}" for k in range(max(0, n - len(ADJ)))]
    for i in range(1, n + 1):
        imp = rng.choice([0, rng.randint(500, 20000)]) if i == n or rng.random() < 0.1 else rng.randint(500, 20000)
        clicks = 0 if imp == 0 else rng.randint(0, imp // 20)
        conv = None if rng.random() < 0.2 else (0 if clicks == 0 else rng.randint(0, max(1, clicks // 5)))
        rows.append([i, names[i - 1], rng.choice(CHANNELS), imp, clicks, Decimal(rng.randint(0, 50000)) / 100, conv])
    if "zero" in traps:
        rows[0][3] = 0
        rows[0][4] = 0
    sql = CAMPAIGNS_DDL + insert("campaigns", ["id", "name", "channel", "impressions", "clicks", "spend",
                                               "conversions"], rows, per_line=2)
    return sql, {"rows": rows}


CAMPAIGN_TRAPS = [("campaigns with no impressions", ("zero",)), ("missing conversions", ()), ("more campaigns", ("zero",))]
