"""12.12 Recipes: Jupyter & a model API."""

from ._common import API_DOCKERFILE, SERVE, P, catalog, docker, dockerfile, files, heredoc, setup, template

M = "12.12"
NB = "quay.io/jupyter/scipy-notebook"
BATCH = SERVE + '''

class Batch(BaseModel):
    texts: list[str]


@app.post("/batch")
def batch(req: Batch):
    return [clf(t)[0] for t in req.texts]
'''


def _api(rng, extra=None):
    fs = {"serve.py": SERVE, "Dockerfile": API_DOCKERFILE}
    fs.update(extra or {})
    return setup(docker("pull=python:3.12-slim"), "#: cwd ~/sentiment", files(fs))


@template(M, "lite")
def jupyter(rng):
    port = rng.choice([8888, 8889])
    s = setup(docker(f"pull={NB}"), "#: cwd ~/notebooks")
    items = [
        ("docker run", f"Start Jupyter Lab from `{NB}` detached as `lab`, on host port {port} (container 8888), with "
                       "the current folder mounted at `/home/jovyan/work`.",
         f'docker run -d --name lab -p {port}:8888 -v "$(pwd)":/home/jovyan/work {NB}',
         {"check": ["containers"]}),
        ("docker logs", f"Jupyter is running as `lab`. Print the login URL with its token from the logs (the first "
                        "line that contains `127.0.0.1:8888/lab?token=`, grep the logs, stderr included).",
         "docker logs lab 2>&1 | grep -m1 '127.0.0.1:8888/lab?token='",
         {"setup": f'docker run -dq --name lab -p {port}:8888 -v "$(pwd)":/home/jovyan/work {NB} >/dev/null'}),
    ]
    return catalog(rng, items, s)


@template(M, "lite")
def build_api(rng):
    return P("Build the sentiment API in this folder as `sentiment-api`, run it detached as `api` with port 8000 "
             "published, wait a few seconds, and call `/health`.", _api(rng),
             "docker build -t sentiment-api .\ndocker run -d -p 8000:8000 --name api sentiment-api\nsleep 8\n"
             "curl -s localhost:8000/health", check=["containers"], post="curl -s localhost:8000/health",
             require=["docker build", "docker run -p|docker run --publish"])


@template(M, "easy")
def predict(rng):
    text = rng.choice(["I love Docker", "This film was terrible", "What a lovely day", "The build is broken again"])
    s = setup(_api(rng), "docker build -q -t sentiment-api . >/dev/null",
              "docker run -dq -p 8000:8000 --name api sentiment-api >/dev/null", "sleep 8")
    return P(f"The API runs as `api` on port 8000. POST `{{\"text\": \"{text}\"}}` to `/predict` as JSON with curl.",
             s, f"curl -X POST localhost:8000/predict -H \"Content-Type: application/json\" -d '{{\"text\": \"{text}\"}}'",
             require=["curl"])


@template(M, "easy")
def api_logs(rng):
    s = setup(_api(rng), "docker build -q -t sentiment-api . >/dev/null",
              "docker run -dq -p 8000:8000 --name api sentiment-api >/dev/null", "sleep 8",
              "curl -s localhost:8000/health >/dev/null", "curl -s localhost:8000/nope >/dev/null")
    items = [
        ("docker logs", "Show the API's request log lines (uvicorn writes them to stderr; keep only lines with "
                        "`HTTP/1.1`).", "docker logs api 2>&1 | grep HTTP/1.1"),
        ("docker logs", "Find the request that got a 404 in `api`'s logs.", "docker logs api 2>&1 | grep ' 404 '"),
    ]
    return catalog(rng, items, s)


@template(M, "medium")
def batch_endpoint(rng):
    texts = rng.sample(["great product", "awful support", "works fine", "terrible latency", "love it"], 3)
    body = '{"texts": [' + ", ".join(f'"{t}"' for t in texts) + "]}"
    s = _api(rng)
    return P("Add a `/batch` endpoint to `serve.py` that takes `{\"texts\": [...]}` and returns one prediction per "
             "text (a `Batch` pydantic model with `texts: list[str]`, and `return [clf(t)[0] for t in req.texts]`). "
             "Rebuild `sentiment-api`, run it detached as `api` on port 8000, and call `/batch` with "
             f"`{body}`.", s,
             heredoc("serve.py", BATCH) + "\ndocker build -t sentiment-api .\ndocker run -d -p 8000:8000 --name api "
                                          f"sentiment-api\nsleep 8\ncurl -s -X POST localhost:8000/batch -H "
                                          f"'Content-Type: application/json' -d '{body}'",
             post=f"curl -s -X POST localhost:8000/batch -H 'Content-Type: application/json' -d '{body}'",
             require=["docker build"])


@template(M, "medium")
def localhost_bind(rng):
    s = _api(rng, {"Dockerfile": API_DOCKERFILE.replace('"--host", "0.0.0.0", ', "")})
    return P("The container runs, but `curl localhost:8000/health` from the host gets *connection reset*: uvicorn "
             "listens only on the container's own 127.0.0.1. Make it listen on all interfaces (`--host 0.0.0.0`), "
             "rebuild `sentiment-api`, run it detached as `api` on port 8000, and call `/health`.", s,
             heredoc("Dockerfile", API_DOCKERFILE) + "\ndocker build -t sentiment-api .\ndocker run -d -p 8000:8000 "
                                                     "--name api sentiment-api\nsleep 8\ncurl -s localhost:8000/health",
             post="curl -s localhost:8000/health", require=["docker build"])


@template(M, "hard")
def offline_weights(rng):
    df = dockerfile("FROM python:3.12-slim", "WORKDIR /app", 'RUN pip install --no-cache-dir fastapi "uvicorn[standard]" '
                    "transformers torch huggingface-hub --extra-index-url https://download.pytorch.org/whl/cpu",
                    "ENV HF_HOME=/models",
                    "RUN huggingface-cli download distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                    "COPY serve.py .", "EXPOSE 8000", 'CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", '
                                                     '"8000"]')
    return P("Containers of this API download the model when they start, so they fail without internet. Bake the "
             "weights into the image at build time: set `HF_HOME=/models` and run `huggingface-cli download "
             "distilbert/distilbert-base-uncased-finetuned-sst-2-english` in the Dockerfile (install `huggingface-hub` "
             "too). Rebuild as `sentiment-api:offline` and run it detached as `api` with **no network** "
             "(`--network none`).", _api(rng),
             heredoc("Dockerfile", df) + "\ndocker build -t sentiment-api:offline .\ndocker run -d --network none "
                                         "--name api sentiment-api:offline\nsleep 3\ndocker logs api 2>&1 | tail -1",
             check=["containers"], post="docker logs api 2>&1 | grep -c 'Application startup complete'",
             require=["docker build"])


@template(M, "hard")
def notebook_as_me(rng):
    s = setup(docker(f"pull={NB}"), "#: cwd ~/analysis")
    return P(f"Run `{NB}` detached as `nb` on port 8888 with the current folder at `/home/jovyan/work`, then create "
             "`/home/jovyan/work/hello.ipynb` from inside it with `docker exec nb touch ...` and check on the host "
             "that the file appeared.", s,
             f'docker run -d --name nb -p 8888:8888 -v "$(pwd)":/home/jovyan/work {NB}\n'
             "docker exec nb touch /home/jovyan/work/hello.ipynb\nls", check=["containers"],
             post="ls; stat -c '%u' hello.ipynb", require=["docker exec"])


@template(M, "challenge")
def production_api(rng):
    df = dockerfile("FROM python:3.12-slim", "WORKDIR /app", 'RUN pip install --no-cache-dir fastapi "uvicorn[standard]" '
                    "transformers torch --extra-index-url https://download.pytorch.org/whl/cpu", "COPY serve.py .",
                    "RUN useradd --create-home api", "USER api", "EXPOSE 8000",
                    "HEALTHCHECK --interval=5s CMD python -c \"import urllib.request; "
                    "urllib.request.urlopen('http://localhost:8000/health')\" || exit 1",
                    'CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]')
    return P("**Ship the API.** Harden the Dockerfile: run as a non-root user `api`, add a HEALTHCHECK every 5s that "
             "requests `http://localhost:8000/health` with Python's urllib, keep the CPU-only torch install. Build "
             "`sentiment-api:1.0`, run it detached as `api` with the port published **only on localhost** "
             "(127.0.0.1:8000) and restart policy `unless-stopped`, and wait until it's healthy.", _api(rng),
             heredoc("Dockerfile", df) + "\ndocker build -t sentiment-api:1.0 .\ndocker run -d --name api --restart "
                                         "unless-stopped -p 127.0.0.1:8000:8000 sentiment-api:1.0\nsleep 15",
             check=["image=sentiment-api:1.0", "containers"], post="curl -s localhost:8000/health",
             require=["docker build"])


@template(M, "challenge")
def compose_api(rng):
    yaml = ("services:\n  api:\n    build: .\n    image: sentiment-api:compose\n    ports:\n      - \"8000:8000\"\n"
            "    volumes:\n      - hf-cache:/root/.cache/huggingface\n    restart: unless-stopped\nvolumes:\n  hf-cache:\n")
    return P("**Compose it.** Put the API in a `compose.yaml`: service `api` built here and tagged "
             "`sentiment-api:compose`, port 8000, the model cache in a named volume `hf-cache` at "
             "`/root/.cache/huggingface` (so restarts don't re-download), restart `unless-stopped`. Start it detached, "
             "then restart the service and call `/predict` with `{\"text\": \"so good\"}`.", _api(rng),
             heredoc("compose.yaml", yaml) + "\ndocker compose up -d\nsleep 8\ndocker compose restart api\nsleep 3\n"
                                             "curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' "
                                             "-d '{\"text\": \"so good\"}'",
             check=["compose=sentiment"], post="docker compose logs api 2>&1 | grep -c 'model.safetensors'",
             require=["docker compose up"])
