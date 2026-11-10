"""12.09 Networking."""

from ._common import P, catalog, docker, files, setup, template

M = "12.09"
PG = "postgres:16"


@template(M, "lite")
def network_commands(rng):
    net = rng.choice(["ml-net", "backend", "app-net"])
    s = setup(docker("pull=ubuntu:24.04"), "docker network create old-net >/dev/null",
              "docker run -dq --name box --network old-net ubuntu:24.04 sleep infinity >/dev/null")
    items = [
        ("docker network create", f"Create a user-defined bridge network `{net}`.", f"docker network create {net}",
         {"check": ["networks"]}),
        ("docker network ls", "List the networks' names only.", "docker network ls --format '{{.Name}}'",
         {"unordered": True}),
        ("docker network inspect", "Print the subnet of `old-net` (`.IPAM.Config`).",
         "docker network inspect -f '{{range .IPAM.Config}}{{.Subnet}}{{end}}' old-net"),
        ("docker network inspect", "Print the names of the containers attached to `old-net`.",
         "docker network inspect -f '{{range .Containers}}{{.Name}}{{end}}' old-net"),
        ("docker network rm", "Remove `old-net` (stop and remove `box` first).",
         "docker rm -f box\ndocker network rm old-net", {"check": ["networks", "containers"]}),
    ]
    return catalog(rng, items, s)


@template(M, "lite")
def by_name(rng):
    return P("Start Postgres 16 detached as `db` on a new network `ml-net` (password `secret`), then from a throwaway "
             "`postgres:16` container on the same network run `psql -h db -U postgres -c 'SELECT 1'` (pass the "
             "password with `-e PGPASSWORD=secret`).", setup(docker(f"pull={PG}")),
             f"docker network create ml-net\ndocker run -d --name db --network ml-net -e POSTGRES_PASSWORD=secret {PG}\n"
             f"sleep 3\ndocker run --rm --network ml-net -e PGPASSWORD=secret {PG} psql -h db -U postgres -c 'SELECT 1'",
             check=["containers", "networks"], require=["docker run --network|docker run --net"])


@template(M, "easy")
def default_bridge(rng):
    s = setup(docker(f"pull={PG},curlimages/curl"), "docker run -dq --name api nginx >/dev/null",
              "docker run -dq --name db -e POSTGRES_PASSWORD=secret postgres:16 >/dev/null", "sleep 3")
    items = [
        ("docker network connect", "`api` and `db` run on the default bridge, where containers can't find each "
                                   "other by name. Create a network `app-net` and connect both containers to it.",
         "docker network create app-net\ndocker network connect app-net api\ndocker network connect app-net db",
         {"check": ["networks"], "post": "docker exec api curl -s -o /dev/null -w '%{http_code}\\n' http://api"}),
        ("docker run --network", "From a throwaway `curlimages/curl` container, fetch `http://api` by name: create a "
                                 "network `web`, connect `api` to it, and run the curl container on it.",
         "docker network create web\ndocker network connect web api\ndocker run --rm --network web curlimages/curl "
         "-s http://api", {"check": ["networks"]}),
    ]
    return catalog(rng, items, setup(s, "docker pull -q nginx >/dev/null"))


@template(M, "easy")
def inspect_ip(rng):
    s = setup(docker("pull=nginx"), "docker network create mesh >/dev/null",
              "docker run -dq --name a --network mesh nginx >/dev/null",
              "docker run -dq --name b --network mesh nginx >/dev/null")
    items = [
        ("docker inspect", "Print container `b`'s IP address on the `mesh` network.",
         "docker inspect -f '{{.NetworkSettings.Networks.mesh.IPAddress}}' b"),
        ("docker exec", "From inside `a`, fetch the page `b` serves using the name `b` (`curl -s http://b`).",
         "docker exec a curl -s http://b"),
        ("docker network inspect", "Print the containers on `mesh` with their IPv4 addresses as `NAME ADDRESS`.",
         "docker network inspect -f '{{range .Containers}}{{.Name}} {{.IPv4Address}}{{println}}{{end}}' mesh",
         {"unordered": True}),
    ]
    return catalog(rng, items, s)


@template(M, "medium")
def localhost_only(rng):
    port = rng.choice([8000, 8080, 8888])
    return P(f"Run nginx detached as `admin` but publish its port 80 **only on localhost** (127.0.0.1) at port {port}, "
             f"so it isn't reachable from your network.", setup(docker("pull=nginx")),
             f"docker run -d --name admin -p 127.0.0.1:{port}:80 nginx", check=["containers"],
             post=f"docker port admin; curl -s localhost:{port} | grep -o '<h1>.*</h1>'",
             require=["docker run -p|docker run --publish"])


@template(M, "medium")
def aliases(rng):
    alias = rng.choice(["database", "pg", "store"])
    return P(f"Create a network `data`, start Postgres 16 detached as `db` on it (password `secret`) with the extra "
             f"network alias `{alias}`, and check from a throwaway `postgres:16` container that `pg_isready -h {alias}` "
             f"works.", setup(docker(f"pull={PG}")),
             f"docker network create data\ndocker run -d --name db --network data --network-alias {alias} -e "
             f"POSTGRES_PASSWORD=secret {PG}\nsleep 3\ndocker run --rm --network data {PG} pg_isready -h {alias}",
             check=["containers", "networks"], require=["docker run --network-alias"])


@template(M, "hard")
def host_service(rng):
    port = rng.choice([9000, 9100, 8765])
    s = setup(docker("pull=curlimages/curl"), "#: cwd ~/site", files({"index.html": "<p>served by the host</p>\n"}),
              f"python3 -m http.server {port} >/dev/null 2>&1 &", "sleep 1")
    return P(f"A web server runs **on the host** at port {port}. From a throwaway `curlimages/curl` container, fetch "
             f"`http://host.docker.internal:{port}/`: on Linux you must map that name to the host gateway "
             f"(`--add-host=host.docker.internal:host-gateway`).", s,
             f"docker run --rm --add-host=host.docker.internal:host-gateway curlimages/curl -s "
             f"http://host.docker.internal:{port}/", require=["docker run --add-host"])


@template(M, "hard")
def internal_net(rng):
    return P("Create an **internal** network `private` (no outside access), start an `ubuntu:24.04` container `vault` "
             "on it running `sleep infinity`, and print the networks `vault` is attached to.",
             setup(docker("pull=ubuntu:24.04")),
             "docker network create --internal private\ndocker run -d --name vault --network private ubuntu:24.04 sleep "
             "infinity\ndocker inspect -f '{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}' vault",
             check=["networks", "containers"], require=["docker network create"])


@template(M, "challenge")
def three_tier(rng):
    port = rng.choice([8080, 8088])
    s = setup(docker(f"pull=nginx,{PG}"), "#: cwd ~/stack",
              files({"proxy/default.conf": "server {\n    listen 80;\n    location /api/ {\n        proxy_pass "
                                           "http://api:80/;\n    }\n    location / {\n        root /usr/share/nginx/"
                                           "html;\n    }\n}\n"}))
    return P(f"**Three tiers, one door.** On a new network `stack`: run Postgres 16 as `db` (password `secret`), nginx "
             f"as `api` (a stand-in backend), and nginx as `proxy` with `./proxy/default.conf` mounted read-only at "
             f"`/etc/nginx/conf.d/default.conf`. Only `proxy` is published, on host port {port}; `/api/` requests go "
             f"through it to `api` by name.", s,
             f"docker network create stack\ndocker run -d --name db --network stack -e POSTGRES_PASSWORD=secret {PG}\n"
             "docker run -d --name api --network stack nginx\n"
             f'docker run -d --name proxy --network stack -p {port}:80 -v "$(pwd)/proxy/default.conf":/etc/nginx/'
             "conf.d/default.conf:ro nginx",
             check=["containers", "networks"], post=f"curl -s localhost:{port}/api/ | grep -o '<h1>.*</h1>'")


@template(M, "challenge")
def debug_connection(rng):
    s = setup(docker(f"pull={PG}"), "docker run -dq --name db -e POSTGRES_PASSWORD=secret postgres:16 >/dev/null",
              "sleep 3")
    return P("**Why can't it connect?** `docker run --rm -e PGPASSWORD=secret postgres:16 psql -h db -U postgres -c "
             "'SELECT 1'` fails with *could not translate host name \"db\"*. Fix the setup (keep `db` running: put it "
             "on a new network `pg-net`) and make that query work from a throwaway container.", s,
             "docker network create pg-net\ndocker network connect pg-net db\n"
             f"docker run --rm --network pg-net -e PGPASSWORD=secret {PG} psql -h db -U postgres -c 'SELECT 1'",
             check=["containers", "networks"], require=["docker network connect|docker run --network"])
