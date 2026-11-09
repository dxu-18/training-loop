"""Small practice repositories, continued (16-30): community lists, a company's apps and infrastructure, a game,
and the git-gym repositories built around one skill each (bisect, conflicts, secrets in history, messy commits,
submodules, LFS, hooks)."""

from __future__ import annotations

import json

from .catalog import LFS, PEOPLE, Lazy, recipe
from .catalog_gen import (GITIGNORES, ci_workflow, go_file, jpg, js_module, license_text, md_doc, paragraph, pick,
                          png, py_function, py_module, py_test, sentence)
from .catalog_light import grow, readme, sub


# ============================================================================ 16. open-learning/awesome-ml

SECTIONS = {
    "Courses": ["Practical Deep Learning - a top-down course", "Linear algebra for ML - short video series",
                "Probabilistic ML - lecture notes"],
    "Libraries": ["PyTorch - tensors and autograd", "scikit-learn - classical ML", "JAX - composable transforms"],
    "Datasets": ["CIFAR-10 - small images", "SQuAD - reading comprehension", "OpenStreetMap extracts"],
    "Tools": ["DVC - data versioning", "Weights & Biases - experiment tracking", "nbstripout - clean notebooks"],
}


@recipe("open-learning", "awesome-ml", "A curated list of machine-learning courses, libraries, datasets and tools",
        language="Markdown", topics=["awesome", "awesome-list", "machine-learning"], stars=9100,
        about="A curated Markdown list grown through many small pull requests, with a CONTRIBUTING guide and a "
              "link-check workflow; one open PR conflicts with main.")
def _awesome_ml(b):
    def render(sections):
        out = ["# Awesome ML", "", "A curated list of resources for learning and doing machine learning.", "",
               "## Contents", ""] + [f"- [{s}](#{s.lower()})" for s in sections] + [""]
        for s, items in sections.items():
            out += [f"## {s}", ""] + [f"- {x}" for x in items] + [""]
        return "\n".join(out)
    secs = {k: v[:1] for k, v in SECTIONS.items()}
    b.commit("Initial list", {"README.md": render(secs), "CONTRIBUTING.md": "# Contributing\n\nOne item per pull "
                              "request. Keep each section in alphabetical order. Describe the item in a few words.\n",
                              "LICENSE": license_text("CC0-1.0", 2024, "open-learning"),
                              "media/banner.png": png(1_250_000)}, by="noor-h")
    b.commit("Add link check workflow", {".github/workflows/links.yml": "name: Links\non: [push, pull_request]\n"
                                         "jobs:\n  check:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: "
                                         "actions/checkout@v4\n      - run: npx markdown-link-check README.md\n"},
             by="noor-h")
    n = 2
    people = ["chloe-m", "ravi-k", "grace-l", "omar-s", "zara-q", "felix-w", "ana-p", "yuki-s"]
    for s, items in SECTIONS.items():
        for item in items[1:]:
            who = pick(b.rng, people)
            br = "add-" + item.split(" ")[0].lower().replace("&", "and")
            b.branch(br)
            secs[s] = secs[s] + [item]
            b.commit(f"Add {item.split(' - ')[0]}", {"README.md": render(secs)}, by=who)
            b.switch("main")
            b.merge_pr(n, br, f"Add {item.split(' - ')[0]}", by="noor-h", author=who, method="squash")
            n += 1
    b.branch("add-papers-section")
    secs2 = dict(secs, Papers=["Attention Is All You Need - transformers"])
    b.commit("Add a Papers section", {"README.md": render(secs2)}, by="ben-o")
    b.switch("main")
    secs["Courses"] = secs["Courses"] + ["Full Stack Deep Learning - deployment"]
    b.commit("Add Full Stack Deep Learning", {"README.md": render(secs)}, by="noor-h")
    b.open_pr(n, "add-papers-section", "Add a Papers section", "ben-o", body="New section with landmark papers.")


# ============================================================================ 17. open-learning/cs-notes

@recipe("open-learning", "cs-notes", "Lecture notes for an undergraduate CS curriculum, in Markdown",
        language="Markdown", topics=["computer-science", "notes", "education"], stars=1320,
        about="Many Markdown notes in topic folders (algorithms/, systems/, ml/, math/) with diagrams: a wide "
              "tree for sparse-checkout and path-limited log practice.")
def _cs_notes(b):
    topics = {"algorithms": ["sorting", "graphs", "dynamic-programming", "hashing"],
              "systems": ["processes", "memory", "filesystems", "networking"],
              "ml": ["linear-models", "trees", "neural-nets"], "math": ["probability", "linear-algebra"]}
    b.commit("Start the notes", {"README.md": "# CS notes\n\nOne folder per course. PRs with fixes are welcome.\n"},
             by="ana-p")
    people = ["ana-p", "elif-k", "kwame-a", "chloe-m"]
    for folder, notes in topics.items():
        files = {}
        for nt in notes:
            files[f"{folder}/{nt}.md"] = md_doc(b.rng, nt.replace("-", " ").title(), [nt.split("-")[0], "examples",
                                                                                     "exercises"], 4)
        files[f"{folder}/img/overview.png"] = png(b.rng.randint(250_000, 600_000))
        b.commit(f"Add {folder} notes", files, by=pick(b.rng, people))
    paths = [f"{f}/{n}.md" for f, ns in topics.items() for n in ns]
    grow(b, 18, people, paths)


# ============================================================================ 18. acme-corp/kanban-board

@recipe("acme-corp", "kanban-board", "A team kanban board: React frontend, Express backend",
        language="JavaScript", topics=["react", "express", "kanban"], stars=230,
        about="A team project with a protected main branch (pull requests and one approval required), several "
              "contributors, open PRs in review, a draft PR, and labelled issues including good first issues.")
def _kanban(b):
    b.commit("Initial commit", {"package.json": json.dumps({"name": "kanban-board", "private": True,
                                                           "workspaces": ["client", "server"],
                                                           "scripts": {"test": "npm test --workspaces"}}, indent=2)
                                + "\n", ".gitignore": GITIGNORES["node"], "README.md": "# Kanban board\n\n"
                                "`npm install && npm run dev`\n"}, by="sokafor")
    b.commit("Server: boards and cards API", {"server/index.js": js_module(b.rng, "Express app.", 3, esm=False),
                                              "server/routes/cards.js": js_module(b.rng, "Card routes.", 4, esm=False)},
             by="sokafor")
    b.commit("Client: board view", {"client/src/Board.jsx": js_module(b.rng, "The board: columns of cards.", 3),
                                    "client/src/Card.jsx": js_module(b.rng, "A single card.", 2),
                                    "client/public/logo.png": png(310_000)}, by="grace-l")
    b.commit("Add CI", {".github/workflows/ci.yml": ci_workflow("node")}, by="sokafor")
    people = ["sokafor", "grace-l", "omar-s", "ana-p", "zara-q"]
    n = 1
    for title, br, who in [("Persist cards in SQLite", "feature/sqlite", "omar-s"),
                           ("Card labels and colours", "feature/labels", "ana-p"),
                           ("Fix card order after reload", "fix/card-order", "zara-q"),
                           ("Keyboard shortcuts", "feature/shortcuts", "grace-l")]:
        b.branch(br)
        grow(b, 2, [who], ["server/routes/cards.js", "client/src/Board.jsx", "client/src/Card.jsx"])
        b.switch("main")
        b.merge_pr(n, br, title, by="sokafor", author=who, reviews=[("grace-l" if who != "grace-l" else "sokafor",
                                                                     "APPROVED", "LGTM")])
        n += 2
    b.commit("Add screenshots", {"docs/board.png": png(1_400_000), "docs/dark-mode.png": png(1_150_000)},
             by="grace-l")
    for num, title, who, labels in [(2, "Cards jump to the top after editing", "kwame-a", ["bug"]),
                                    (4, "Dark mode", "zara-q", ["enhancement"]),
                                    (6, "Add a CONTRIBUTING guide", "sokafor", ["documentation", "good first issue"]),
                                    (8, "Typo on the empty-board message", "ana-p", ["good first issue"]),
                                    (10, "Archive old cards", "omar-s", ["enhancement"])]:
        b.issue(num, title, who, labels=labels, closed=num == 2)
    b.branch("feature/drag-columns")
    grow(b, 2, ["omar-s"], ["client/src/Board.jsx"])
    b.switch("main")
    b.open_pr(11, "feature/drag-columns", "Drag and drop columns", "omar-s",
              reviews=[("grace-l", "CHANGES_REQUESTED", "Columns flicker while dragging.")])
    b.branch("feature/dark-mode")
    b.commit("Dark mode: CSS variables", {"client/src/theme.css": ":root { --bg: #fff; }\n@media (prefers-color-"
                                          "scheme: dark) { :root { --bg: #111; } }\n"}, by="zara-q")
    b.switch("main")
    b.open_pr(12, "feature/dark-mode", "Dark mode", "zara-q", draft=True, body="Closes #4")
    b.repo.protected["main"] = {"pr": True, "reviews": 1}


# ============================================================================ 19. acme-corp/express-api

@recipe("acme-corp", "express-api", "REST API for the Acme store, built with Express",
        language="JavaScript", topics=["nodejs", "express", "rest-api"], stars=140,
        about="A Node REST API with a CI workflow; the feature/pagination branch has a failing test, so its pull "
              "request shows a failed check until it is fixed.")
def _express(b):
    b.commit("Initial commit", {"package.json": json.dumps({"name": "acme-api", "version": "1.0.0",
                                                           "scripts": {"start": "node src/index.js",
                                                                       "test": "node --test test/"}}, indent=2) + "\n",
                                "src/index.js": js_module(b.rng, "Express entry point.", 2, esm=False),
                                "src/routes/products.js": js_module(b.rng, "Product routes.", 4, esm=False),
                                "src/routes/orders.js": js_module(b.rng, "Order routes.", 3, esm=False),
                                "test/products.test.js": "const test = require('node:test');\nconst assert = "
                                                         "require('node:assert');\n\ntest('lists products', () => {\n"
                                                         "  assert.ok(true);\n});\n",
                                ".gitignore": GITIGNORES["node"], ".github/workflows/ci.yml": ci_workflow("node"),
                                "README.md": "# acme-api\n\n`npm start`, then GET /products.\n",
                                "docs/openapi.yaml": "openapi: 3.0.0\ninfo:\n  title: Acme API\n  version: 1.0.0\n"},
             by="omar-s")
    b.commit("Add API diagrams", {"docs/architecture.png": png(1_320_000)}, by="omar-s")
    grow(b, 6, ["omar-s", "sokafor", "ravi-k"], ["src/routes/products.js", "src/routes/orders.js", "src/index.js"])
    b.tag("v1.0.0", "First deploy")
    b.branch("feature/pagination")
    b.edit("src/routes/products.js", lambda t: t + "\nconst PAGE_SIZE = 20;\n", "Paginate /products", by="ravi-k")
    b.commit("Test pagination", {"test/pagination.test.js": "const test = require('node:test');\nconst assert = "
                                 "require('node:assert');\n\ntest('second page starts at item 21', () => {\n"
                                 "  assert.fail('pagination returns page 1 twice');\n});\n"}, by="ravi-k")
    b.switch("main")
    b.open_pr(5, "feature/pagination", "Paginate the products endpoint", "ravi-k")
    b.issue(3, "GET /products returns 5000 items", "sokafor", labels=["performance"])


# ============================================================================ 20. acme-corp/terraform-infra

@recipe("acme-corp", "terraform-infra", "Terraform for Acme's AWS networking and services",
        language="HCL", topics=["terraform", "aws", "infrastructure-as-code"], stars=35,
        about="Terraform modules and environments. An early commit accidentally added terraform.tfstate (with "
              "made-up resource ids and a fake password); a later commit deleted it, but it is still in history.")
def _terraform(b):
    b.commit("Initial modules", {"main.tf": 'terraform {\n  required_version = ">= 1.6"\n}\n\nmodule "vpc" {\n  '
                                            'source = "./modules/vpc"\n  cidr   = var.vpc_cidr\n}\n',
                                 "variables.tf": 'variable "vpc_cidr" {\n  type    = string\n  default = '
                                                 '"10.0.0.0/16"\n}\n',
                                 "modules/vpc/main.tf": 'resource "aws_vpc" "this" {\n  cidr_block = var.cidr\n}\n',
                                 "modules/vpc/variables.tf": 'variable "cidr" { type = string }\n',
                                 "README.md": "# terraform-infra\n\n`terraform init && terraform plan`\n"},
             by="kwame-a")
    b.commit("Add dev environment", {"envs/dev/main.tf": 'module "stack" {\n  source = "../../"\n}\n',
                                     "terraform.tfstate": json.dumps({"version": 4, "resources": [
                                         {"type": "aws_db_instance", "name": "main", "instances": [
                                             {"attributes": {"id": "db-EXAMPLEFAKE0001", "username": "admin",
                                                             "password": "FAKE-not-a-real-password-1234"}}]}]},
                                         indent=2) + "\n"}, by="kwame-a")
    b.commit("Remove state file, add .gitignore", {".gitignore": GITIGNORES["terraform"]},
             delete=["terraform.tfstate"], by="sokafor")
    b.commit("Add prod environment", {"envs/prod/main.tf": 'module "stack" {\n  source = "../../"\n}\n'},
             by="kwame-a")
    b.commit("Add architecture diagram", {"docs/network.png": png(1_650_000)}, by="sokafor")
    grow(b, 4, ["kwame-a", "sokafor"], ["README.md"])
    b.issue(2, "State file was committed in history", "sokafor", "It's deleted now but still reachable from old "
            "commits. Should we rewrite history?", labels=["security"])


# ============================================================================ 21. acme-corp/payments-service

@recipe("acme-corp", "payments-service", "Payment processing microservice in Go",
        language="Go", topics=["go", "microservice", "payments"], stars=60,
        about="A Go service with release branches (release/1.4, release/1.5) next to main. A security fix "
              "landed on main and must be cherry-picked onto the release branches; tags v1.4.0, v1.4.1, v1.5.0.")
def _payments(b):
    b.commit("Initial service", {"go.mod": "module github.com/acme-corp/payments-service\n\ngo 1.22\n",
                                 "cmd/server/main.go": 'package main\n\nfunc main() {\n\tserve(":8080")\n}\n',
                                 "internal/charge/charge.go": go_file(b.rng, "charge", "creates and captures "
                                                                     "charges.", 3),
                                 "internal/refund/refund.go": go_file(b.rng, "refund", "issues refunds.", 2),
                                 "README.md": "# payments-service\n\nRelease branches: `release/X.Y`. Fixes land on "
                                              "main first, then get cherry-picked.\n"}, by="dalvarez")
    grow(b, 4, ["dalvarez", "jberg"], ["internal/charge/charge.go", "internal/refund/refund.go"])
    b.tag("v1.4.0", "v1.4.0")
    b.branch("release/1.4")
    b.commit("Fix rounding of refund amounts", {"internal/refund/refund.go": b.read("internal/refund/refund.go") +
                                                "\n// amounts are rounded half-even\n"}, by="jberg")
    b.tag("v1.4.1", "v1.4.1")
    b.switch("main")
    grow(b, 3, ["dalvarez", "jberg", "elif-k"], ["internal/charge/charge.go"])
    b.commit("Add docs", {"docs/flows.png": png(1_050_000), "docs/states.png": png(890_000)}, by="elif-k")
    b.tag("v1.5.0", "v1.5.0")
    b.branch("release/1.5")
    b.switch("main")
    grow(b, 2, ["dalvarez"], ["internal/charge/charge.go"])
    b.commit("Validate webhook signatures with constant-time comparison\n\nComparing with == leaks timing "
             "information an attacker can use to forge signatures.",
             {"internal/webhook/verify.go": 'package webhook\n\nimport "crypto/hmac"\n\n// Verify reports whether '
                                            'sig is the HMAC of body.\nfunc Verify(sig, mac []byte) bool {\n\treturn '
                                            'hmac.Equal(sig, mac)\n}\n'}, by="elif-k")
    grow(b, 2, ["jberg"], ["internal/refund/refund.go"])
    b.issue(9, "Backport the webhook timing fix to 1.4 and 1.5", "elif-k", labels=["security", "backport"])


# ============================================================================ 22. pixel-forge/snake-game

SNAKE_PY = '''"""Snake in pygame. Arrow keys to move, Esc to quit."""

import random

import pygame

CELL = 20
W, H = 32, 24
SPEED = 10


def new_food(snake):
    while True:
        p = (random.randrange(W), random.randrange(H))
        if p not in snake:
            return p


def main():
    pygame.init()
    screen = pygame.display.set_mode((W * CELL, H * CELL))
    clock = pygame.time.Clock()
    snake = [(W // 2, H // 2)]
    direction = (1, 0)
    food = new_food(snake)
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return
        head = (snake[0][0] + direction[0], snake[0][1] + direction[1])
        snake.insert(0, head)
        if head == food:
            food = new_food(snake)
        else:
            snake.pop()
        clock.tick(SPEED)


if __name__ == "__main__":
    main()
'''


@recipe("pixel-forge", "snake-game", "Classic Snake in Python with pygame",
        language="Python", topics=["game", "pygame", "beginner"], stars=75,
        about="A beginner-sized game: one Python file, sprites and sounds, and a high-scores branch that was "
              "never merged.")
def _snake(b):
    b.commit("Snake moves and eats", {"snake.py": SNAKE_PY, "requirements.txt": "pygame==2.6.1\n",
                                      ".gitignore": GITIGNORES["python"], "README.md": "# Snake\n\n`pip install -r "
                                                                                       "requirements.txt && python "
                                                                                       "snake.py`\n"}, by="yuki-s")
    b.commit("Add sprites and sounds", {"assets/head.png": png(96_000), "assets/body.png": png(88_000),
                                        "assets/apple.png": png(120_000), "sounds/eat.wav": Lazy(410_000, b"RIFF"),
                                        "sounds/game_over.wav": Lazy(980_000, b"RIFF"),
                                        "sounds/music.ogg": Lazy(1_700_000, b"OggS")}, by="ben-o")
    b.edit("snake.py", lambda t: sub(t, "SPEED = 10", "SPEED = 12"), "Speed up a little", by="yuki-s")
    b.edit("snake.py", lambda t: sub(t, "        snake.insert(0, head)\n",
                                     "        if head in snake or not (0 <= head[0] < W and 0 <= head[1] < H):\n"
                                     "            return\n        snake.insert(0, head)\n"),
           "End the game on collision", by="yuki-s")
    b.branch("high-scores")
    b.commit("Save high scores to scores.json", {"scores.py": py_module(b.rng, "Load and save high scores.", 2,
                                                                        ["load_scores", "save_score"], ("json",))},
             by="ben-o")
    b.switch("main")
    b.edit("README.md", lambda t: t + "\nArrow keys move. Esc quits.\n", "Document controls", by="yuki-s")


# ============================================================================ 23. git-gym/calculator

CALC = '''"""A tiny calculator library."""


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def div(a, b):
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a / b
'''

CALC_TEST = '''"""Run: python3 test_calc.py (exit status 0 = all good)."""

import sys

from calc import add, div, mul, sub

failures = 0
for got, want, what in [(add(2, 3), 5, "add"), (sub(5, 3), 2, "sub"), (mul(4, 3), 12, "mul"),
                        (div(7, 2), 3.5, "div")]:
    if got != want:
        print(f"FAIL {what}: got {got!r}, want {want!r}")
        failures += 1
print("ok" if not failures else f"{failures} failed")
sys.exit(1 if failures else 0)
'''


@recipe("git-gym", "calculator", "A calculator library with a bug hiding somewhere in its history",
        language="Python", topics=["git-practice", "bisect"], stars=15,
        about="Forty commits of small changes to a calculator; one of them quietly breaks div() (7 / 2 gives 3). "
              "v1.0 is known good. test_calc.py exits non-zero when a result is wrong: made for git bisect run.")
def _calculator(b):
    b.commit("Initial calculator", {"calc.py": CALC, "test_calc.py": CALC_TEST, "README.md": "# calculator\n\n"
                                    "`python3 test_calc.py` checks the four operations.\n"}, by="mira-chen")
    b.tag("v1.0", "Known good release")
    docs = {"README.md"}
    people = ["mira-chen", "tomas-ruiz", "priya-n", "jberg"]
    for i in range(38):
        if i == 23:
            b.edit("calc.py", lambda t: sub(t, "    return a / b", "    return a // b"),
                   "Simplify div", by="tomas-ruiz")
            continue
        if i % 3 == 0:
            b.edit("calc.py", lambda t: t + "\n\n" + py_function(b.rng), pick(b.rng, ["Add helper", "Add utility",
                                                                                       "More helpers"]),
                   by=pick(b.rng, people))
        else:
            b.edit("README.md", lambda t: t + sentence(b.rng) + "\n", pick(b.rng, ["Update README", "Docs", "Clarify "
                                                                                    "usage", "Typo"]),
                   by=pick(b.rng, people))
    b.commit("Add logo", {"logo.png": png(1_200_000)}, by="mira-chen")
    b.issue(1, "div(7, 2) returns 3", "ben-o", "It used to return 3.5 in v1.0.", labels=["bug", "regression"])


# ============================================================================ 24. git-gym/recipe-book

CHILI = """# Chili

Serves 4.

## Ingredients

- 500 g beef mince
- 1 onion
- 2 cloves garlic
- 1 tin kidney beans
- 1 tin tomatoes
- 1 tsp chili powder

## Method

1. Brown the mince.
2. Add onion and garlic, cook 5 minutes.
3. Add everything else and simmer for 30 minutes.
"""


@recipe("git-gym", "recipe-book", "A shared recipe book: three cooks, three branches, one chili",
        language="Markdown", topics=["git-practice", "merge-conflicts"], stars=9,
        about="Two branches (spicy, vegan) and main all change the same lines of recipes/chili.md, so merging "
              "them conflicts; other recipes merge cleanly.")
def _recipes(b):
    b.commit("Start the recipe book", {"README.md": "# Recipe book\n\nOne Markdown file per recipe.\n",
                                       "recipes/chili.md": CHILI,
                                       "recipes/pancakes.md": "# Pancakes\n\n- 200 g flour\n- 2 eggs\n- 300 ml milk\n",
                                       "photos/chili.jpg": jpg(1_350_000), "photos/pancakes.jpg": jpg(980_000)},
             by="chloe-m")
    b.branch("spicy")
    b.edit("recipes/chili.md", lambda t: sub(t, "- 1 tsp chili powder", "- 1 tbsp chili powder\n- 2 fresh "
                                                                        "jalapenos"),
           "Make the chili properly spicy", by="dalvarez")
    b.edit("recipes/chili.md", lambda t: sub(t, "3. Add everything else and simmer for 30 minutes.",
                                             "3. Add everything else and simmer for 45 minutes."),
           "Simmer longer", by="dalvarez")
    b.switch("main")
    b.branch("vegan")
    b.edit("recipes/chili.md", lambda t: sub(sub(t, "- 500 g beef mince", "- 400 g lentils"),
                                             "1. Brown the mince.", "1. Rinse the lentils."),
           "Vegan chili: lentils instead of beef", by="noor-h")
    b.commit("Add hummus", {"recipes/hummus.md": "# Hummus\n\n- 1 tin chickpeas\n- 2 tbsp tahini\n- 1 lemon\n"},
             by="noor-h")
    b.switch("main")
    b.edit("recipes/chili.md", lambda t: sub(sub(t, "Serves 4.", "Serves 6."), "- 1 tsp chili powder",
                                             "- 1 tsp chili powder\n- 1 tsp cumin"),
           "Bigger batch, add cumin", by="chloe-m")
    b.commit("Add pancake photo credits", {"photos/CREDITS.md": "Photos by the recipe authors.\n"}, by="chloe-m")


# ============================================================================ 25. git-gym/leaky-app

@recipe("git-gym", "leaky-app", "A small Flask app whose history leaks secrets and a database dump",
        language="Python", topics=["git-practice", "security", "filter-repo"], stars=7,
        about="Commit 3 added a .env with fake API keys and commit 5 a 3 MB SQL dump; later commits delete both, "
              "but they stay in history until it is rewritten (git filter-repo), and the keys are 'rotated'.")
def _leaky(b):
    b.commit("Initial app", {"app.py": py_module(b.rng, "The web app.", 3, ["create_app", "index"], ("os",)),
                             "requirements.txt": "flask==3.0.3\npython-dotenv==1.0.1\n",
                             "README.md": "# leaky-app\n\nCopy `.env.example` to `.env` and fill it in.\n"},
             by="sokafor")
    b.commit("Read settings from the environment", {"config.py": 'import os\n\nAPI_KEY = os.environ["API_KEY"]\n'
                                                    'DB_URL = os.environ.get("DB_URL", "sqlite:///app.db")\n'},
             by="sokafor")
    b.commit("Add local settings", {".env": "API_KEY=sk_test_FAKE_51HxAmPlEkEy0000000000\n"
                                            "DB_URL=postgres://admin:FAKEpassword@db.internal:5432/app\n"
                                            "AWS_SECRET_ACCESS_KEY=FAKEfakeFAKEfakeFAKEfakeFAKEfake0000\n"},
             by="sokafor")
    b.commit("Add templates", {"templates/index.html": "<h1>Leaky app</h1>\n"}, by="ana-p")
    b.commit("Add database dump for local testing", {"data/dump.sql": Lazy(3_100_000, b"-- PostgreSQL database dump\n"
                                                                                       b"\nSET statement_timeout = 0;"
                                                                                       b"\n")}, by="ana-p")
    b.commit("Remove .env, add .env.example and .gitignore", {".env.example": "API_KEY=\nDB_URL=\n"
                                                                              "AWS_SECRET_ACCESS_KEY=\n",
                                                              ".gitignore": GITIGNORES["python"] + "data/*.sql\n"},
             delete=[".env"], by="sokafor")
    b.commit("Remove the database dump", delete=["data/dump.sql"], by="ana-p")
    grow(b, 4, ["sokafor", "ana-p"], ["app.py", "README.md"])
    b.issue(1, "Secrets are still in the git history", "elif-k", "Deleting .env in a new commit doesn't remove it "
            "from old commits. We need to rewrite history and rotate the keys.", labels=["security"])


# ============================================================================ 26. git-gym/messy-history

@recipe("git-gym", "messy-history", "A feature branch full of 'wip' and 'fix typo' commits",
        language="Python", topics=["git-practice", "rebase"], stars=4,
        about="feature/search has eight messy commits (wip, typo fixes, a debug print added then removed) and "
              "main has moved on: squash and reword with interactive rebase, then rebase onto main.")
def _messy(b):
    b.commit("Initial commit", {"app.py": py_module(b.rng, "The application.", 3, ["main"], ("sys",)),
                                "README.md": "# messy-history\n", "docs/screenshot.png": png(1_100_000)},
             by="jberg")
    grow(b, 3, ["jberg"], ["app.py"])
    b.branch("feature/search")
    steps = [("wip", {"search.py": "def search(items, q):\n    return [x for x in items if q in x]\n"}),
             ("wip 2", {"search.py": "def search(items, q):\n    q = q.lower()\n    return [x for x in items if q in "
                                     "x.lower()]\n"}),
             ("fix typo", {"README.md": "# messy-history\n\nSearch is case-insensitve.\n"}),
             ("add debug print", {"search.py": "def search(items, q):\n    q = q.lower()\n    print('DEBUG', q)\n"
                                               "    return [x for x in items if q in x.lower()]\n"}),
             ("tests", {"test_search.py": "from search import search\n\nassert search(['Apple', 'pear'], 'app') == "
                                          "['Apple']\nprint('ok')\n"}),
             ("remove debug print", {"search.py": "def search(items, q):\n    q = q.lower()\n    return [x for x in "
                                                  "items if q in x.lower()]\n"}),
             ("fix typo again", {"README.md": "# messy-history\n\nSearch is case-insensitive.\n"}),
             ("oops forgot docstring", {"search.py": 'def search(items, q):\n    """Case-insensitive substring '
                                                     'search."""\n    q = q.lower()\n    return [x for x in items if '
                                                     'q in x.lower()]\n'})]
    for msg, files in steps:
        b.commit(msg, files, by="jberg", hours=(0, 3))
    b.switch("main")
    b.edit("app.py", lambda t: t + "\n\nVERSION = '1.1'\n", "Bump version", by="mira-chen")
    b.commit("Add CONTRIBUTING", {"CONTRIBUTING.md": "Squash your fixups before opening a PR.\n"}, by="mira-chen")


# ============================================================================ 27-28. git-gym/shared-lib, app-with-submodule

@recipe("git-gym", "shared-lib", "Utility code shared by several Acme apps",
        language="Python", topics=["git-practice", "submodules", "library"], stars=6,
        about="A small library used as a submodule by git-gym/app-with-submodule; tags v1.0.0 and v1.1.0 (the "
              "app pins v1.0.0, so updating the submodule is a practice step).")
def _shared_lib(b):
    b.commit("Initial library", {"shared/__init__.py": "", "shared/text.py": py_module(
        b.rng, "Text helpers.", 3, ["slugify", "truncate"], ()), "README.md": "# shared-lib\n",
        "assets/icons.png": png(1_050_000)}, by="kwame-a")
    b.tag("v1.0.0", "v1.0.0", by="kwame-a")
    b.commit("Add date helpers", {"shared/dates.py": py_module(b.rng, "Date helpers.", 2, ["parse_date"],
                                                               ("datetime",))}, by="kwame-a")
    b.tag("v1.1.0", "v1.1.0", by="kwame-a")


@recipe("git-gym", "app-with-submodule", "An app that vendors git-gym/shared-lib as a git submodule",
        language="Python", topics=["git-practice", "submodules"], stars=5,
        about="lib/shared is a submodule pointing at git-gym/shared-lib (pinned to v1.0.0) over SSH: clone with "
              "--recurse-submodules or run git submodule update --init, then move it to v1.1.0.")
def _app_submodule(b):
    lib = b.gh.get("git-gym", "shared-lib")
    pinned = b.db.tag(lib.refs["refs/tags/v1.0.0"]).object
    b.commit("Initial app", {"app.py": "from lib.shared.shared.text import slugify\n\nprint(slugify('Hello World'))\n",
                             "README.md": "# app-with-submodule\n\n```\ngit clone --recurse-submodules git@github.com"
                                          ":git-gym/app-with-submodule.git\n```\n",
                             "static/banner.png": png(1_300_000)}, by="kwame-a")
    b.commit("Add shared-lib as a submodule", {
        ".gitmodules": '[submodule "lib/shared"]\n\tpath = lib/shared\n\turl = git@github.com:git-gym/shared-lib.git\n',
        "lib/shared": ("160000", pinned)}, by="kwame-a")
    grow(b, 2, ["kwame-a"], ["README.md"])


# ============================================================================ 29. git-gym/lfs-assets

@recipe("git-gym", "lfs-assets", "Design assets stored with Git LFS",
        language="", topics=["git-practice", "git-lfs"], stars=8,
        about="PSD and PNG files tracked with Git LFS (.gitattributes): without git-lfs installed a clone has "
              "small pointer files; after `git lfs install` and `git lfs pull` the real images appear.")
def _lfs_assets(b):
    b.commit("Track design files with LFS", {".gitattributes": "*.psd filter=lfs diff=lfs merge=lfs -text\n"
                                                               "*.png filter=lfs diff=lfs merge=lfs -text\n",
                                             "README.md": "# lfs-assets\n\nNeeds Git LFS: `sudo apt install git-lfs "
                                                          "&& git lfs install`.\n"}, by="chloe-m")
    b.commit("Add logo sources", {"logo/logo.psd": LFS(1_800_000, b"8BPS"), "logo/logo.png": LFS(420_000,
                                                                                                b"\x89PNG\r\n")},
             by="chloe-m")
    b.commit("Add icons", {f"icons/{n}.png": LFS(60_000 + i * 7000, b"\x89PNG\r\n") for i, n in
                           enumerate(["home", "search", "settings", "user", "cart"])}, by="ana-p")
    b.commit("New logo colours", {"logo/logo.png": LFS(431_000, b"\x89PNG\r\n", "logo-v2")}, by="chloe-m")
    b.commit("Add style guide", {"STYLE.md": md_doc(b.rng, "Style guide", ["colour", "spacing", "type"])},
             by="chloe-m")


# ============================================================================ 30. git-gym/hooks-demo

@recipe("git-gym", "hooks-demo", "A Python project that checks every commit with pre-commit",
        language="Python", topics=["git-practice", "pre-commit", "ruff"], stars=11,
        about="Has a .pre-commit-config.yaml (whitespace, end-of-file, large files, ruff, ruff-format) and code "
              "that doesn't pass it yet: install pre-commit, run it on all files, fix, commit.")
def _hooks_demo(b):
    b.commit("Initial project", {"src/report.py": "import os\nimport sys\n\n\ndef build_report(rows):   \n"
                                                  "    out = []\n    for r in rows:\n        out.append(str(r))\n"
                                                  "    return '\\n'.join(out)\n\n\n",
                                 "src/util.py": "import json\n\n\ndef load(path):\n    if path == None:\n"
                                                "        return {}\n    with open(path) as f:\n        return "
                                                "json.load(f)",
                                 "README.md": "# hooks-demo\n\n```\npip install pre-commit\npre-commit install\n"
                                              "pre-commit run --all-files\n```\n",
                                 "docs/workflow.png": png(1_150_000)}, by="felix-w")
    b.commit("Add pre-commit config", {".pre-commit-config.yaml": (
        "repos:\n  - repo: https://github.com/pre-commit/pre-commit-hooks\n    rev: v5.0.0\n    hooks:\n"
        "      - id: trailing-whitespace\n      - id: end-of-file-fixer\n      - id: check-added-large-files\n"
        "  - repo: https://github.com/astral-sh/ruff-pre-commit\n    rev: v0.6.9\n    hooks:\n      - id: ruff\n"
        "      - id: ruff-format\n")}, by="felix-w")
    b.commit("Add CONTRIBUTING", {"CONTRIBUTING.md": "Run `pre-commit install` once after cloning. Commits that fail "
                                                     "the hooks are rejected.\n"}, by="felix-w")
