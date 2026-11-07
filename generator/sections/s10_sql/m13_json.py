"""10.13 JSON & arrays."""

from __future__ import annotations

from ._common import template, drill, shop_db, Q
from . import _data as D


def ev_db(rng, **kw):
    sql, f = D.events(rng, **kw)
    return sql, f, D.hidden(D.events, rng, f, D.EVENT_TRAPS, k=3, n_users=kw.get("n_users", 5))


def runs_db(rng, **kw):
    sql, f = D.runs(rng, **kw)
    return sql, f, D.hidden(D.runs, rng, f, D.RUNS_TRAPS, k=3, n_runs=kw.get("n_runs", 8))


@template("10.13", "lite")
def json_arrows(rng):
    db, f, hid = ev_db(rng)
    items = [
        ("Show each event's `event_id` and `page`: the payload's `page` as text (->>).",
         "SELECT event_id, payload->>'page' AS page FROM events;", {"require": ["->>"]}),
        ("Show each event's `event_id` and `device`: the payload's `device` as JSON (->).",
         "SELECT event_id, payload->'device' AS device FROM events;", {"require": ["->"]}),
        ("Show `event_id` of the events on the `/pricing` page.",
         "SELECT event_id FROM events WHERE payload->>'page' = '/pricing';"),
    ]
    return drill(rng, items, db, hid)


@template("10.13", "lite")
def array_basics(rng):
    db, f, hid = runs_db(rng)
    items = [
        ("Show each run's `run_id` and `first_tag`: the first element of `tags` (arrays are 1-indexed).",
         "SELECT run_id, tags[1] AS first_tag FROM runs;"),
        ("Show each run's `run_id` and `n_tags` (ARRAY_LENGTH of tags along dimension 1; NULL for an empty array).",
         "SELECT run_id, ARRAY_LENGTH(tags, 1) AS n_tags FROM runs;", {"require": ["ARRAY_LENGTH("]}),
        ("Show one row with `arr`: the array [1, 2, 3], and `second`: its second element.",
         "SELECT ARRAY[1, 2, 3] AS arr, (ARRAY[1, 2, 3])[2] AS second;"),
    ]
    return drill(rng, items, db, hid)


@template("10.13", "easy")
def json_paths_contains(rng):
    db, f, hid = ev_db(rng)
    items = [
        ("Show `event_id` and `amount` (payload amount as text, #>> path) of the purchase events.",
         "SELECT event_id, payload #>> '{amount}' AS amount FROM events WHERE event_type = 'purchase';",
         {"require": ["#>>"]}),
        ("Show `event_id` of the events whose payload contains `{\"device\": \"mobile\"}` (@>).",
         "SELECT event_id FROM events WHERE payload @> '{\"device\": \"mobile\"}';", {"require": ["@>"]}),
        ("Show `event_id` of the events whose payload has a `page` key (?).",
         "SELECT event_id FROM events WHERE payload ? 'page';", {"require": ["?"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.13", "easy")
def json_build(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show one `doc` per user: a JSONB object with keys `id` and `name` (JSONB_BUILD_OBJECT).",
         "SELECT JSONB_BUILD_OBJECT('id', id, 'name', name) AS doc FROM users;", {"require": ["JSONB_BUILD_OBJECT("]}),
        ("Show the `id` of the orders whose status is one of an array of statuses ('paid', 'pending') with = ANY.",
         "SELECT id FROM orders WHERE status = ANY(ARRAY['paid', 'pending']);", {"require": ["ANY"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.13", "medium")
def json_flatten(rng):
    db, f, hid = ev_db(rng)
    items = [
        ("Flatten the purchased items: one row per (`event_id`, `item`) from the payload's `items` array "
         "(JSONB_ARRAY_ELEMENTS_TEXT).",
         "SELECT e.event_id, t.item FROM events e, JSONB_ARRAY_ELEMENTS_TEXT(e.payload->'items') AS t(item);",
         {"require": ["JSONB_ARRAY_ELEMENTS_TEXT("]}),
        ("Count how often each item was bought: `item` and `n`.",
         "SELECT t.item, COUNT(*) AS n FROM events e, JSONB_ARRAY_ELEMENTS_TEXT(e.payload->'items') AS t(item) GROUP "
         "BY t.item;"),
    ]
    return drill(rng, items, db, hid)


@template("10.13", "medium")
def json_agg_unnest(rng):
    db, f, hid = shop_db(rng)
    runs_sql, rf = D.runs(rng)
    items = [
        ("For each user with orders show `user_id` and `orders`: a JSONB array of {id, amount} objects in id order.",
         "SELECT user_id, JSONB_AGG(JSONB_BUILD_OBJECT('id', id, 'amount', amount) ORDER BY id) AS orders FROM orders "
         "GROUP BY user_id;", {"require": ["JSONB_AGG("]}),
        ("For each user with orders show `user_id` and `statuses`: the array of distinct statuses, sorted.",
         "SELECT user_id, ARRAY_AGG(DISTINCT status ORDER BY status) AS statuses FROM orders GROUP BY user_id;",
         {"require": ["ARRAY_AGG("]}),
    ]
    return drill(rng, items, db, hid)


@template("10.13", "medium")
def array_unnest(rng):
    db, f, hid = runs_db(rng)
    items = [
        ("Count runs per tag: `tag` (UNNEST of tags) and `runs`.",
         "SELECT tag, COUNT(*) AS runs FROM runs, UNNEST(tags) AS tag GROUP BY tag;", {"require": ["UNNEST("]}),
        ("Show the `run_id` of the runs tagged `baseline` (= ANY(tags)).",
         "SELECT run_id FROM runs WHERE 'baseline' = ANY(tags);", {"require": ["ANY"]}),
        ("Show the `run_id` of the runs tagged both `sweep` and `fp16` (@> on arrays).",
         "SELECT run_id FROM runs WHERE tags @> ARRAY['sweep', 'fp16'];"),
    ]
    from ...core import pick
    prompt, sol, *rest = pick(rng, items)
    opts = rest[0] if rest else {}
    return Q(prompt, sol, db, hid, **opts)     # no answer on this data: the template retries with other data


@template("10.13", "hard")
def json_typed(rng):
    db, f, hid = runs_db(rng)
    lr = rng.choice(["0.001", "0.0001", "0.01"])
    items = [
        (f"Show the `run_id` of the runs whose config has lr = {lr} (cast config->>'lr' to NUMERIC).",
         f"SELECT run_id FROM runs WHERE (config->>'lr')::NUMERIC = {lr};", {"require": ["->>"]}),
        ("For each optimizer in config show `optimizer` and `avg_dropout` (config dropout as numeric, averaged, "
         "rounded to 3 decimals).",
         "SELECT config->>'optimizer' AS optimizer, ROUND(AVG((config->>'dropout')::NUMERIC), 3) AS avg_dropout FROM "
         "runs GROUP BY 1;"),
        ("Show each run's `run_id` and `config2`: the config with `warmup` set to 0 (JSONB_SET).",
         "SELECT run_id, JSONB_SET(config, '{warmup}', '0') AS config2 FROM runs;", {"require": ["JSONB_SET("]}),
        ("Show each run's `run_id` and `keys`: the config keys, comma-separated in alphabetical order.",
         "SELECT run_id, (SELECT STRING_AGG(k, ',' ORDER BY k) FROM JSONB_OBJECT_KEYS(config) AS k) AS keys FROM runs;"),
    ]
    from ...core import pick
    prompt, sol, *rest = pick(rng, items)
    opts = rest[0] if rest else {}
    return Q(prompt, sol, db, hid, **opts)     # no answer on this data: the template retries with other data


@template("10.13", "hard")
def json_gin(rng):
    db, f, hid = ev_db(rng)
    items = [
        ("Create a GIN index `idx_events_payload` on events(payload), then count the events whose payload contains "
         "{\"device\": \"desktop\"} as `n`.",
         "CREATE INDEX idx_events_payload ON events USING GIN (payload);\nSELECT COUNT(*) AS n FROM events WHERE "
         "payload @> '{\"device\": \"desktop\"}';",
         {"check": "state", "tables": ["events"], "compare": ["indexes"], "also": "query"}),
        ("Show one row per (`event_id`, `key`, `value`) for every key of each purchase payload (JSONB_EACH_TEXT).",
         "SELECT e.event_id, kv.key, kv.value FROM events e, JSONB_EACH_TEXT(e.payload) AS kv WHERE e.event_type = "
         "'purchase';", {"require": ["JSONB_EACH_TEXT("]}),
    ]
    return drill(rng, items, db, hid)


@template("10.13", "challenge")
def json_sweep(rng):
    db, f, hid = runs_db(rng, n_runs=12)
    sol = ("WITH best AS (SELECT r.run_id, r.model, (r.config->>'lr')::NUMERIC AS lr, m.loss, ROW_NUMBER() OVER "
           "(PARTITION BY r.model ORDER BY m.loss, r.run_id) AS rn FROM runs r JOIN metrics m ON m.run_id = r.run_id "
           "AND m.split = 'val' AND m.step = 300) SELECT model, run_id, lr, loss FROM best WHERE rn = 1 ORDER BY "
           "model;")
    return Q("Sweep post-mortem: for each `model`, the run with the lowest validation loss at step 300 (ties: lower "
             "run_id): `model`, `run_id`, `lr` (from the JSONB config, as NUMERIC) and `loss`. Runs without a step-300 "
             "validation metric don't count. By model.", sol, db, hid, order=True)


@template("10.13", "challenge")
def json_tag_pairs(rng):
    db, f, hid = runs_db(rng, n_runs=12)
    sol = ("SELECT a.tag AS tag_a, b.tag AS tag_b, COUNT(*) AS runs FROM runs r, UNNEST(r.tags) AS a(tag), "
           "UNNEST(r.tags) AS b(tag) WHERE a.tag < b.tag GROUP BY 1, 2 ORDER BY runs DESC, tag_a, tag_b;")
    return Q("Which tags are used together? For every pair of different tags on the same run, show `tag_a` < "
             "`tag_b` and `runs` (how many runs carry both). Most common first, then by tag_a, tag_b.", sol, db, hid,
             order=True)
