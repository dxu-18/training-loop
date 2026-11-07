"""10.07 String functions."""

from __future__ import annotations

import random

from ._common import template, drill, shop_db, Q
from . import _data as D

DOMAINS = ["gmail.com", "uni.edu", "lab.ai", "corp.io", "mail.de"]


def contacts_db(rng, n=7, traps=(), like=None):
    traps = set(traps)
    rows = []
    for i in range(1, n + 1):
        first = rng.choice(D.NAMES)
        last = rng.choice(["Lovelace", "Turing", "Hopper", "Knuth", "Liskov", "Ng", "Li", "Bengio", "Hinton"])
        email = f"{first.lower()}.{last.lower()}@{rng.choice(DOMAINS)}"
        if i % 3 == 0:
            email = email.upper() if "case" in traps else email.capitalize()
        digits = "".join(str(rng.randrange(10)) for _ in range(10))
        style = rng.choice(["({}) {}-{}", "{}-{}-{}", "{}.{}.{}", "+1 {} {} {}"])
        phone = style.format(digits[:3], digits[3:6], digits[6:])
        code = f"{rng.choice(['ml', 'cv', 'nlp'])}-{rng.randint(1, 999):03d}-{rng.choice(['a', 'b', 'x'])}"
        note = rng.choice(["  trailing  ", "ok", "  lead", "VIP ", None])
        rows.append([i, f"{first} {last}", email, phone, code, note])
    sql = ("CREATE TABLE contacts (\n  id INTEGER PRIMARY KEY,\n  full_name TEXT NOT NULL,\n  email TEXT NOT NULL,\n"
           "  phone TEXT NOT NULL,\n  code TEXT NOT NULL,\n  note TEXT\n);\n" +
           D.insert("contacts", ["id", "full_name", "email", "phone", "code", "note"], rows, per_line=1))
    return sql, {"rows": rows}


TRAPS = [("upper-case emails", ("case",)), ("other phone styles", ()), ("more contacts", ("case",))]


def cdb(rng):
    sql, f = contacts_db(rng)
    return sql, f, D.hidden(contacts_db, rng, f, TRAPS, k=3, n=7)


@template("10.07", "lite")
def str_basic(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show each user's `name`, `upper_name` (UPPER) and `len` (LENGTH).",
         "SELECT name, UPPER(name) AS upper_name, LENGTH(name) AS len FROM users;", {"require": ["UPPER(", "LENGTH("]}),
        ("Show each user's `tag`: the lower-case name, a dash and the country (NULL when the country is unknown).",
         "SELECT LOWER(name) || '-' || country AS tag FROM users;", {"require": ["LOWER("]}),
    ]
    return drill(rng, items, db, hid)


@template("10.07", "lite")
def str_concat(rng):
    db, f, hid = shop_db(rng, traps=("null_country",))
    items = [
        ("Show each user's `label`: name and country separated by a slash, with CONCAT (a missing country gives "
         "just `name/`).", "SELECT CONCAT(name, '/', country) AS label FROM users;", {"require": ["CONCAT("]}),
        ("Show each user's `path`: `users`, the country and the name joined by `/`, skipping a missing country "
         "(CONCAT_WS).", "SELECT CONCAT_WS('/', 'users', country, name) AS path FROM users;",
         {"require": ["CONCAT_WS("]}),
    ]
    return drill(rng, items, db, hid)


@template("10.07", "easy")
def str_slices(rng):
    db, f, hid = cdb(rng)
    items = [
        ("Show each contact's `id` and `initials`: the first letter of the first name and of the last name.",
         "SELECT id, LEFT(full_name, 1) || SUBSTRING(full_name FROM POSITION(' ' IN full_name) + 1 FOR 1) AS initials "
         "FROM contacts;"),
        ("Show each contact's `id` and `first_name` (everything before the space).",
         "SELECT id, LEFT(full_name, POSITION(' ' IN full_name) - 1) AS first_name FROM contacts;",
         {"require": ["POSITION(|STRPOS("]}),
        ("Show each contact's `id` and `note_clean`: the note without leading and trailing spaces.",
         "SELECT id, TRIM(note) AS note_clean FROM contacts;", {"require": ["TRIM("]}),
        ("Show each contact's `id` and `code_tail`: the last character of the code.",
         "SELECT id, RIGHT(code, 1) AS code_tail FROM contacts;", {"require": ["RIGHT("]}),
    ]
    return drill(rng, items, db, hid)


@template("10.07", "easy")
def str_replace_pos(rng):
    db, f, hid = cdb(rng)
    items = [
        ("Show each contact's `id` and `code_us`: the code with `-` replaced by `_`.",
         "SELECT id, REPLACE(code, '-', '_') AS code_us FROM contacts;", {"require": ["REPLACE("]}),
        ("Show each contact's `id` and `at_pos`: the position of `@` in the email.",
         "SELECT id, STRPOS(email, '@') AS at_pos FROM contacts;", {"require": ["STRPOS(|POSITION("]}),
    ]
    return drill(rng, items, db, hid)


@template("10.07", "medium")
def str_split_pad(rng):
    db, f, hid = cdb(rng)
    items = [
        ("Show each contact's `id` and `number`: the middle part of the code (between the dashes) as text.",
         "SELECT id, SPLIT_PART(code, '-', 2) AS number FROM contacts;", {"require": ["SPLIT_PART("]}),
        ("Show each contact's `padded_id`: the id zero-padded to 6 digits.",
         "SELECT LPAD(id::TEXT, 6, '0') AS padded_id FROM contacts;", {"require": ["LPAD("]}),
        ("Show each contact's `id` and `domain`: the email's part after `@`, in lower case.",
         "SELECT id, LOWER(SPLIT_PART(email, '@', 2)) AS domain FROM contacts;"),
        ("Show each contact's `id` and `pretty`: the full name with each word capitalised (INITCAP of the lower "
         "case).", "SELECT id, INITCAP(LOWER(full_name)) AS pretty FROM contacts;", {"require": ["INITCAP("]}),
    ]
    return drill(rng, items, db, hid)


@template("10.07", "medium")
def str_format(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show one line per user, `line`, like `Ana has 2 orders`, with FORMAT (users without orders: 0).",
         "SELECT FORMAT('%s has %s orders', u.name, COUNT(o.id)) AS line FROM users u LEFT JOIN orders o ON o.user_id "
         "= u.id GROUP BY u.id, u.name;", {"require": ["FORMAT("]}),
        ("Show each order's `receipt`: `#<id> <amount>` with the id padded with zeros to 4 digits.",
         "SELECT '#' || LPAD(id::TEXT, 4, '0') || ' ' || amount AS receipt FROM orders;"),
    ]
    return drill(rng, items, db, hid)


@template("10.07", "hard")
def str_regex(rng):
    db, f, hid = cdb(rng)
    dom = rng.choice(sorted({r[2].split("@")[1].lower() for r in f["rows"]}))
    esc = dom.replace(".", "\\.")
    items = [
        ("Show each contact's `id` and `digits`: the phone with every non-digit removed.",
         "SELECT id, REGEXP_REPLACE(phone, '[^0-9]', '', 'g') AS digits FROM contacts;",
         {"require": ["REGEXP_REPLACE("]}),
        ("Show the `id` of the contacts whose code matches `letters-3 digits-letter` exactly (regex with ~).",
         "SELECT id FROM contacts WHERE code ~ '^[a-z]+-[0-9]{3}-[a-z]$';", {"require": ["~"]}),
        ("Show each contact's `id` and `num`: the first run of digits in the code (SUBSTRING ... FROM regex).",
         "SELECT id, SUBSTRING(code FROM '[0-9]+') AS num FROM contacts;"),
        (f"Show the `id` of the contacts with an email at `{dom}`, ignoring case (~*).",
         f"SELECT id FROM contacts WHERE email ~* '@{esc}$';", {"require": ["~*"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.07", "hard")
def str_domains(rng):
    db, f, hid = cdb(rng)
    items = [
        ("For each email domain (lower case) show `domain` and `n` contacts.",
         "SELECT LOWER(SPLIT_PART(email, '@', 2)) AS domain, COUNT(*) AS n FROM contacts GROUP BY 1;"),
        ("Show each contact's `id` and `phone_fmt`: the 10 digits of the phone as `xxx-xxx-xxxx` (drop a leading "
         "country code 1 if there are 11 digits).",
         "SELECT id, REGEXP_REPLACE(RIGHT(REGEXP_REPLACE(phone, '\\D', '', 'g'), 10), '(\\d{3})(\\d{3})(\\d{4})', "
         "'\\1-\\2-\\3') AS phone_fmt FROM contacts;"),
    ]
    return drill(rng, items, db, hid)


@template("10.07", "challenge")
def str_model_ids(rng):
    sql, f = D.model_registry(rng)
    hid = D.hidden(D.model_registry, rng, f, D.MODEL_TRAPS, k=3)
    sol = ("SELECT SPLIT_PART(name, '-', 1) AS family, SUBSTRING(name FROM '-([0-9.]+)b') AS size_b, "
           "COALESCE(NULLIF(SUBSTRING(name FROM '-[0-9.]+b-(.*)$'), ''), 'base') AS variant, COUNT(*) AS n FROM models "
           "GROUP BY 1, 2, 3 ORDER BY family, size_b, variant;")
    return Q("Model names look like `family-SIZEb` or `family-SIZEb-variant...`. Parse them and count models per "
             "(`family`, `size_b`, `variant`): family is the part before the first dash, size_b the number before "
             "`b` (as text), variant everything after the size (or `base`). By family, size_b, variant.",
             sol, sql, hid, order=True)


@template("10.07", "challenge")
def str_cleanup(rng):
    db, f, hid = cdb(rng)
    sol = ("SELECT id, INITCAP(LOWER(SPLIT_PART(full_name, ' ', 1))) AS first_name, LOWER(email) AS email, "
           "REGEXP_REPLACE(phone, '\\D', '', 'g') AS phone, UPPER(SPLIT_PART(code, '-', 1)) || '-' || "
           "LPAD(SPLIT_PART(code, '-', 2), 4, '0') AS code, NULLIF(TRIM(note), '') AS note FROM contacts ORDER BY id;")
    return Q("Clean the contacts for import, by id: `id`; `first_name` (first word, capitalised); `email` (lower "
             "case); `phone` (digits only); `code` (first part upper-cased, a dash, the number padded to 4 digits; "
             "drop the last part); `note` (trimmed, NULL when nothing is left).", sol, db, hid, order=True)
