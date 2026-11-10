"""12.10 Docker Compose."""

from ._common import API_DOCKERFILE, SERVE, P, catalog, docker, files, heredoc, setup, template

M = "12.10"
PG = "postgres:16"
COMPOSE = """services:
  web:
    image: nginx
    ports:
      - "{port}:80"
  db:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: secret
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata:
"""


def _stack(rng, port=8080, project="shop"):
    return setup(docker(f"pull=nginx,{PG}"), f"#: cwd ~/{project}",
                 files({"compose.yaml": COMPOSE.format(port=port)}))


@template(M, "lite")
def up_down(rng):
    port = rng.choice([8080, 8081, 9000])
    s = _stack(rng, port)
    items = [
        ("docker compose up", "Start the whole stack in `compose.yaml` in the background.", "docker compose up -d",
         {"require": ["docker compose up"]}),
        ("docker compose down", "The stack is running. Stop and remove it (keep the volume).",
         "docker compose down", {"setup": "docker compose up -d >/dev/null 2>&1"}),
        ("docker compose down", "The stack is running. Stop and remove it **and** its volumes.",
         "docker compose down -v", {"setup": "docker compose up -d >/dev/null 2>&1",
                                    "require": ["docker compose down"]}),
        ("docker compose up", "Start only the `web` service, in the background.", "docker compose up -d web"),
    ]
    return catalog(rng, items, s, check=["containers", "volumes", "networks"])


@template(M, "lite")
def ps_logs(rng):
    s = setup(_stack(rng), "docker compose up -d >/dev/null 2>&1", "sleep 3", "curl -s localhost:8080 >/dev/null")
    items = [
        ("docker compose ps", "List the stack's running services' names (`--services`).",
         "docker compose ps --services", {"unordered": True}),
        ("docker compose logs", "Show the `db` service's logs.", "docker compose logs db"),
        ("docker compose logs", "Show only the last line of the `web` service's logs.", "docker compose logs --tail 1 "
                                                                                       "web"),
        ("docker compose exec", "Run `psql -U postgres -c 'SELECT 2'` inside the running `db` service (no TTY, `-T`).",
         "docker compose exec -T db psql -U postgres -c 'SELECT 2'"),
        ("docker compose config", "Print the resolved compose file's service names (`config --services`).",
         "docker compose config --services", {"unordered": True}),
    ]
    return catalog(rng, items, s)


@template(M, "easy")
def write_compose(rng):
    port = rng.choice([8000, 8080])
    s = setup(docker(f"pull={PG},nginx"), "#: cwd ~/blog", files({"html/index.html": "<h1>Blog</h1>\n"}))
    yaml = (f"services:\n  web:\n    image: nginx\n    ports:\n      - \"{port}:80\"\n    volumes:\n      - ./html:/usr/"
            f"share/nginx/html:ro\n  db:\n    image: postgres:16\n    environment:\n      POSTGRES_PASSWORD: secret\n")
    return P(f"Write `compose.yaml` with two services: `web` (nginx, host port {port} -> 80, `./html` mounted "
             f"read-only at `/usr/share/nginx/html`) and `db` (`postgres:16` with `POSTGRES_PASSWORD=secret`). Start it "
             f"detached.", s, heredoc("compose.yaml", yaml) + "\ndocker compose up -d", check=["containers"],
             post=f"curl -s localhost:{port}", require=["docker compose up"])


@template(M, "easy")
def run_tests(rng):
    s = setup(docker("pull=python:3.12-slim"), "#: cwd ~/lib",
              files({"compose.yaml": "services:\n  app:\n    image: python:3.12-slim\n    working_dir: /src\n    "
                                     "volumes:\n      - .:/src\n",
                     "test_math.py": "def add(a, b):\n    return a + b\n\n\nassert add(2, 3) == 5\nprint('2 passed')\n"}))
    return P("Run the test file in a one-off container of the `app` service: `python test_math.py`, removing the "
             "container afterwards (`compose run --rm`).", s, "docker compose run --rm app python test_math.py",
             post="docker ps -aq | wc -l", require=["docker compose run"])


@template(M, "medium")
def healthy_dependency(rng):
    s = setup(docker(f"pull={PG},nginx"), "#: cwd ~/app",
              files({"compose.yaml": "services:\n  api:\n    image: nginx\n    ports:\n      - \"8000:80\"\n    "
                                     "depends_on:\n      - db\n  db:\n    image: postgres:16\n    environment:\n      "
                                     "POSTGRES_PASSWORD: secret\n"}))
    yaml = ("services:\n  api:\n    image: nginx\n    ports:\n      - \"8000:80\"\n    depends_on:\n      db:\n        "
            "condition: service_healthy\n  db:\n    image: postgres:16\n    environment:\n      POSTGRES_PASSWORD: "
            "secret\n    healthcheck:\n      test: [\"CMD-SHELL\", \"pg_isready -U postgres\"]\n      interval: 5s\n"
            "      retries: 5\n")
    return P("The API starts before Postgres is ready. Give `db` a health check (`pg_isready -U postgres`, every 5s, 5 "
             "retries) and make `api` wait until `db` is **healthy** (`depends_on` with `condition: "
             "service_healthy`). Then bring the stack up detached.", s,
             heredoc("compose.yaml", yaml) + "\ndocker compose up -d", check=["containers"],
             post="docker compose ps --format '{{.Service}} {{.Health}}' | sort", require=["docker compose up"])


@template(M, "medium")
def env_and_volumes(rng):
    model = rng.choice(["distilbert", "resnet18", "gpt2"])
    s = setup(docker("pull=python:3.12-slim"), "#: cwd ~/svc",
              files({".env": f"MODEL_NAME={model}\nLOG_LEVEL=info\n",
                     "show.py": "import os\nprint(os.environ.get('MODEL_NAME'), os.environ.get('LOG_LEVEL'), "
                                "os.environ.get('WORKERS'))\n"}))
    yaml = ("services:\n  worker:\n    image: python:3.12-slim\n    working_dir: /src\n    command: python show.py\n"
            "    env_file: .env\n    environment:\n      - WORKERS=4\n    volumes:\n      - .:/src\n      - "
            "cache:/root/.cache\nvolumes:\n  cache:\n")
    return P("Write `compose.yaml` with one service `worker`: `python:3.12-slim`, working in `/src` with this folder "
             "mounted there, command `python show.py`, variables from `.env` plus `WORKERS=4`, and a named volume "
             "`cache` at `/root/.cache`. Run it in the foreground once with `docker compose up` (it exits by itself).",
             s, heredoc("compose.yaml", yaml) + "\ndocker compose up", check=["volumes"],
             post="docker compose logs --no-log-prefix worker", require=["docker compose up"])


@template(M, "hard")
def build_service(rng):
    s = setup(docker("pull=python:3.12-slim"), "#: cwd ~/ml-api", files({"serve.py": SERVE, "Dockerfile": API_DOCKERFILE}))
    yaml = ("services:\n  api:\n    build: .\n    image: ml-api:dev\n    ports:\n      - \"8000:8000\"\n    volumes:\n"
            "      - hf-cache:/root/.cache/huggingface\nvolumes:\n  hf-cache:\n")
    return P("Write `compose.yaml` for the model API in this folder: a service `api` **built** from the Dockerfile here "
             "and tagged `ml-api:dev`, published on 8000, with a named volume `hf-cache` at "
             "`/root/.cache/huggingface`. Bring it up detached (building the image), wait a few seconds, and call "
             "`/health`.", s, heredoc("compose.yaml", yaml) + "\ndocker compose up -d --build\nsleep 8\n"
                                "curl -s localhost:8000/health",
             check=["containers", "volumes"], post="curl -s localhost:8000/health", require=["docker compose up"])


@template(M, "hard")
def fix_compose(rng):
    which = rng.choice(["indent", "key", "volume"])
    good = COMPOSE.format(port=8080)
    bad = {"indent": good.replace("    image: postgres:16", "   image: postgres:16"),
           "key": good.replace("    image: nginx", "    imagee: nginx"),
           "volume": good.replace("volumes:\n  pgdata:\n", "")}[which]
    s = setup(docker(f"pull=nginx,{PG}"), "#: cwd ~/shop", files({"compose.yaml": bad}))
    return P("`docker compose up -d` rejects this `compose.yaml`. Use `docker compose config` to find the mistake, fix "
             "the file, and start the stack detached.", s,
             "docker compose config\n" + heredoc("compose.yaml", good) + "\ndocker compose up -d",
             check=["containers", "volumes"], require=["docker compose up"])


@template(M, "challenge")
def ml_stack(rng):
    s = setup(docker(f"pull={PG},python:3.12-slim"), "#: cwd ~/ml-stack",
              files({"serve.py": SERVE, "Dockerfile": API_DOCKERFILE}))
    yaml = ("services:\n  api:\n    build: .\n    ports:\n      - \"8000:8000\"\n    environment:\n      - "
            "DATABASE_URL=postgresql://postgres:secret@db:5432/postgres\n    volumes:\n      - hf-cache:/root/.cache/"
            "huggingface\n    depends_on:\n      db:\n        condition: service_healthy\n  db:\n    image: postgres:16\n"
            "    environment:\n      POSTGRES_PASSWORD: secret\n    volumes:\n      - pgdata:/var/lib/postgresql/data\n"
            "    healthcheck:\n      test: [\"CMD-SHELL\", \"pg_isready -U postgres\"]\n      interval: 5s\n      "
            "retries: 5\nvolumes:\n  hf-cache:\n  pgdata:\n")
    return P("**The guide's stack.** Write `compose.yaml`: `api` built from this folder, published on 8000, with "
             "`DATABASE_URL=postgresql://postgres:secret@db:5432/postgres`, the `hf-cache` volume at "
             "`/root/.cache/huggingface`, starting only once `db` is healthy; `db` is `postgres:16` (password "
             "`secret`) with its data in a `pgdata` volume and a `pg_isready -U postgres` health check every 5s. "
             "Bring it up detached and call `/predict` with `{\"text\": \"I love Docker\"}`.", s,
             heredoc("compose.yaml", yaml) + "\ndocker compose up -d\nsleep 8\ncurl -s -X POST localhost:8000/predict "
                                             "-H 'Content-Type: application/json' -d '{\"text\": \"I love Docker\"}'",
             check=["compose=ml-stack"], post="curl -s localhost:8000/health", require=["docker compose up"])


@template(M, "challenge")
def scale_down_clean(rng):
    s = setup(_stack(rng), "docker compose up -d >/dev/null 2>&1", "sleep 2",
              "docker compose exec -T db psql -U postgres -c 'CREATE TABLE keep (id int)' >/dev/null")
    return P("**Restart without losing data.** Recreate the whole stack from scratch (`down`, then `up -d`) **without** "
             "deleting the database volume, and show that the table `keep` survived (`\\dt` through `docker compose "
             "exec -T db psql -U postgres`).", s,
             "docker compose down\ndocker compose up -d\nsleep 3\ndocker compose exec -T db psql -U postgres -c '\\dt'",
             check=["containers", "volumes"], require=["docker compose down", "docker compose up"])
