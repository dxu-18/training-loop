"""The five medium practice repositories (about 100 MB each). Their bulk is generated binary content (model
checkpoints, datasets, game assets, vendored archives, firmware) that stays a recipe until read; the histories
are long enough for shallow clones, sparse checkouts, path-limited logs and bisecting to matter."""

from __future__ import annotations

from .catalog import LFS, PEOPLE, Lazy, recipe
from .catalog_gen import (GITIGNORES, c_file, ci_workflow, csv_text, go_file, ident, jpg, js_module, license_text,
                          md_doc, paragraph, pick, png, py_function, py_module, sentence)
from .catalog_light import grow, readme, sub

MB = 1024 * 1024


# ============================================================================ 31. mini-ml/resnet-cifar

@recipe("mini-ml", "resnet-cifar", "ResNet training on CIFAR-10 with released checkpoints (Git LFS)",
        language="Python", topics=["deep-learning", "computer-vision", "pytorch", "git-lfs"], stars=1890, kind="medium",
        about="An image-classification training repo (~100 MB): model and training code, configs, and model "
              "checkpoints stored with Git LFS (*.pt), plus sample images in git. Clone without git-lfs to see "
              "pointer files; tags mark the checkpoints behind each reported accuracy.")
def _resnet(b):
    b.commit("Initial commit", {
        "README.md": readme("resnet-cifar", "ResNet-18/34 on CIFAR-10 with reproducible configs.", [
            ("Train", "```\npython train.py --config configs/resnet18.yaml\n```"),
            ("Checkpoints", "Stored with Git LFS. Run `git lfs install && git lfs pull` after cloning.")]),
        ".gitignore": GITIGNORES["ml"], ".gitattributes": "*.pt filter=lfs diff=lfs merge=lfs -text\n"
                                                          "*.pth filter=lfs diff=lfs merge=lfs -text\n",
        "requirements.txt": "torch==2.4.1\ntorchvision==0.19.1\npyyaml\ntqdm\n",
        "LICENSE": license_text("MIT", 2024, "mini-ml")}, by="aiko-t")
    b.commit("Add ResNet model", {"models/resnet.py": py_module(b.rng, "ResNet-18 and ResNet-34 for 32x32 inputs.",
                                                                5, ["make_layer", "resnet18", "resnet34"], ("torch",)),
                                  "models/__init__.py": "from .resnet import resnet18, resnet34\n"}, by="aiko-t")
    b.commit("Add training loop and configs", {
        "train.py": py_module(b.rng, "Train on CIFAR-10. Usage: python train.py --config CONFIG", 6,
                              ["parse_args", "build_loaders", "train_epoch", "evaluate", "save_checkpoint"],
                              ("argparse", "os", "yaml", "torch")),
        "configs/resnet18.yaml": "model: resnet18\nepochs: 30\nbatch_size: 128\nlr: 0.1\nweight_decay: 5.0e-4\n",
        "configs/resnet34.yaml": "model: resnet34\nepochs: 40\nbatch_size: 128\nlr: 0.1\nweight_decay: 5.0e-4\n"},
        by="aiko-t")
    b.commit("Add sample images for the README", {f"samples/{c}_{i}.png": png(180_000 + i * 11_000)
                                                  for c in ("cat", "ship", "truck", "frog") for i in range(4)},
             by="yuki-s")
    people = ["aiko-t", "yuki-s", "ravi-k", "grace-l"]
    grow(b, 10, people, ["train.py", "models/resnet.py", "README.md"])
    for ep, acc in ((10, 88.4), (20, 91.9), (30, 93.1)):
        b.commit(f"Add resnet18 checkpoint at epoch {ep} ({acc}% top-1)",
                 {f"checkpoints/resnet18_epoch{ep}.pt": LFS(int(22.5 * MB), b"PK\x03\x04", f"r18-{ep}"),
                  "RESULTS.md": f"| model | epoch | top-1 |\n|---|---|---|\n| resnet18 | {ep} | {acc} |\n"},
                 by="aiko-t")
        b.tag(f"r18-e{ep}", f"resnet18 epoch {ep}: {acc}% top-1", by="aiko-t")
        grow(b, 4, people, ["train.py", "models/resnet.py"])
    b.commit("Add resnet34 checkpoint (94.0% top-1)", {"checkpoints/resnet34_final.pt": LFS(int(21.8 * MB),
                                                                                            b"PK\x03\x04", "r34")},
             by="ravi-k")
    b.tag("v1.0", "Checkpoints for the paper", by="aiko-t")
    b.release("v1.0", "Paper checkpoints", "resnet18 93.1%, resnet34 94.0%. Download with git lfs pull.",
              by="aiko-t")
    b.branch("experiment/mixup")
    b.edit("train.py", lambda t: t + "\n\nMIXUP_ALPHA = 0.2\n", "Try mixup (alpha 0.2)", by="grace-l")
    b.switch("main")
    b.open_pr(7, "experiment/mixup", "Mixup augmentation", "grace-l", draft=True)
    b.issue(3, "Checkpoints are 130-byte text files after cloning", "ben-o", "Opening resnet18_epoch30.pt fails.",
            labels=["question"], comments=[("aiko-t", "Install git-lfs, then `git lfs pull`.")], closed=True)


# ============================================================================ 32. openmaps/city-transit-data

@recipe("openmaps", "city-transit-data", "Open transit data for a mid-sized city: stops, routes and trips",
        language="Python", topics=["open-data", "gtfs", "transit", "datasets"], stars=420, kind="medium",
        about="A data repository (~100 MB) that committed large Parquet files straight into git and replaced "
              "them monthly, so its history is heavy: practise shallow clones (--depth 1), sparse checkout of "
              "one folder, and stripping big blobs from history.")
def _transit(b):
    def stop_row(rng, i):
        return [f"S{i:05d}", f"{pick(rng, ['Main', 'Oak', 'River', 'Station', 'Market', 'Hill'])} St & "
                             f"{rng.randint(1, 99)}th", round(52.3 + rng.random() * 0.2, 6),
                round(4.8 + rng.random() * 0.3, 6), rng.randint(0, 1)]
    b.commit("Initial import of the GTFS feed", {
        "README.md": readme("city-transit-data", "Stops, routes and trips for the city's buses and trams, updated "
                            "monthly.", [("Layout", "- `gtfs/`: the raw feed (CSV)\n- `parquet/`: monthly trip "
                                                    "tables\n- `scripts/`: converters")]),
        "gtfs/stops.txt": csv_text(b.rng, ["stop_id", "stop_name", "stop_lat", "stop_lon", "wheelchair_boarding"],
                                   stop_row, 6000),
        "gtfs/routes.txt": csv_text(b.rng, ["route_id", "route_short_name", "route_type"],
                                    lambda rng, i: [f"R{i}", str(i + 1), pick(rng, [0, 3])], 120),
        "scripts/convert.py": py_module(b.rng, "Convert the GTFS text files to Parquet.", 4,
                                        ["read_feed", "to_parquet"], ("csv", "sys")),
        "LICENSE": license_text("ODbL-1.0", 2024, "openmaps")}, by="kwame-a")
    months = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"]
    for i, m in enumerate(months):
        b.commit(f"Update trips for {m}", {"parquet/trips.parquet": Lazy(int((12 + i) * MB), b"PAR1", f"trips-{m}"),
                                           "parquet/MANIFEST": f"trips.parquet {m}\n"}, by="kwame-a")
        grow(b, 3, ["kwame-a", "elif-k", "omar-s"], ["scripts/convert.py", "README.md"])
    b.commit("Add shapes for tram lines", {"gtfs/shapes.parquet": Lazy(int(9.5 * MB), b"PAR1", "shapes")},
             by="elif-k")
    b.commit("Add city map", {"docs/map.png": png(int(3.2 * MB))}, by="omar-s")
    b.tag("2024-06", "June 2024 release", by="kwame-a")
    b.issue(12, "Cloning takes forever", "zara-q", "The repo is huge because old trips.parquet versions are in "
            "history. Could we move them to LFS or drop old ones?", labels=["discussion"])


# ============================================================================ 33. pixel-forge/dungeon-crawler

@recipe("pixel-forge", "dungeon-crawler", "A 2D roguelike made with Godot; art and audio in Git LFS",
        language="GDScript", topics=["game", "godot", "roguelike", "git-lfs"], stars=2600, kind="medium",
        about="A game project (~100 MB) with scenes and scripts in git and sprites, tilesets and music in Git LFS; "
              "a team of five, feature branches, an open boss-fight PR and release tags.")
def _dungeon(b):
    b.commit("Godot project", {"project.godot": "[application]\nconfig/name=\"Dungeon Crawler\"\nrun/main_scene="
                                                "\"res://scenes/main.tscn\"\n",
                               ".gitattributes": "*.png filter=lfs diff=lfs merge=lfs -text\n*.ogg filter=lfs "
                                                 "diff=lfs merge=lfs -text\n*.wav filter=lfs diff=lfs merge=lfs -text\n",
                               ".gitignore": ".godot/\n*.import\nexport/\n",
                               "README.md": "# Dungeon Crawler\n\nOpen in Godot 4.3. Assets need Git LFS.\n"},
             by="yuki-s")
    scripts = ["player", "enemy", "dungeon_gen", "inventory", "hud", "loot", "save_game"]
    for s in scripts:
        b.commit(f"Add {s} script", {f"scripts/{s}.gd": "extends Node2D\n\n" + "\n".join(
            f"func {ident(b.rng)}(delta):\n\tpass\n" for _ in range(4)),
            f"scenes/{s}.tscn": f"[gd_scene format=3]\n\n[node name=\"{s}\" type=\"Node2D\"]\n"}, by="yuki-s")
    people = ["yuki-s", "ben-o", "chloe-m", "felix-w", "zara-q"]
    sprites = {f"assets/sprites/{n}.png": LFS(int(b.rng.uniform(0.8, 2.4) * MB), b"\x89PNG\r\n", f"spr-{n}")
               for n in ("hero", "skeleton", "slime", "bat", "chest", "door", "tiles_cave", "tiles_crypt")}
    b.commit("Add sprites and tilesets", sprites, by="chloe-m")
    music = {f"assets/music/{n}.ogg": LFS(int(b.rng.uniform(7, 12) * MB), b"OggS", f"mus-{n}")
             for n in ("title", "cave", "crypt", "boss", "victory", "shop")}
    b.commit("Add music", music, by="felix-w")
    sfx = {f"assets/sfx/{n}.wav": LFS(int(b.rng.uniform(0.3, 1.2) * MB), b"RIFF", f"sfx-{n}")
           for n in ("swing", "hit", "pickup", "door", "death", "levelup")}
    b.commit("Add sound effects", sfx, by="felix-w")
    paths = [f"scripts/{s}.gd" for s in scripts]
    for v in ("0.1.0", "0.2.0", "0.3.0"):
        for k in range(5):
            br = f"feature/{pick(b.rng, ['traps', 'shops', 'minimap', 'fog', 'quests', 'pets', 'runes'])}-{v}-{k}"
            who = pick(b.rng, people)
            b.branch(br)
            for _ in range(3):
                p = pick(b.rng, paths)
                b.edit(p, lambda t: t + f"\nfunc {ident(b.rng)}(delta):\n\tpass\n", f"{br.split('/')[1]}: "
                                                                                     f"{sentence(b.rng)[:50]}", by=who)
            b.switch("main")
            b.merge(br, f"Merge branch '{br}'", by="yuki-s", delete=True)
        b.commit(f"New art for {v}", {f"assets/sprites/boss_{v}.png": LFS(int(1.6 * MB), b"\x89PNG\r\n",
                                                                          f"boss-{v}")}, by="chloe-m")
        b.tag(f"v{v}", f"Dungeon Crawler {v}", by="yuki-s")
        b.release(f"v{v}", f"v{v}", sentence(b.rng), by="yuki-s")
    b.branch("feature/boss-fight")
    b.commit("Boss fight: phase one", {"scripts/boss.gd": "extends Node2D\n\nvar phase = 1\n"}, by="ben-o")
    b.commit("Boss music cue", {"assets/music/boss_phase2.ogg": LFS(int(7.2 * MB), b"OggS", "boss2")}, by="felix-w")
    b.switch("main")
    b.open_pr(42, "feature/boss-fight", "Boss fight", "ben-o", reviews=[("yuki-s", "COMMENTED", "Phase two?")])


# ============================================================================ 34. acme-corp/monorepo

SERVICES = {"auth": "go", "billing": "go", "search": "python", "notifications": "node", "gateway": "go"}


@recipe("acme-corp", "monorepo", "Acme's monorepo: services, web app, ML pipelines, infra and docs",
        language="Go", topics=["monorepo", "microservices", "sparse-checkout"], stars=310, kind="medium",
        about="A wide, busy monorepo (~100 MB, hundreds of commits, a dozen people): services/<name> in Go, "
              "Python and Node, web/, ml/, infra/, docs/ and vendored third_party archives. Made for sparse "
              "checkout, shallow and partial clones, CODEOWNERS and path-limited history.")
def _monorepo(b):
    files = {"README.md": readme("Acme monorepo", "Everything Acme ships, in one repository.", [
        ("Layout", "- services/: backend services\n- web/: customer web app\n- ml/: training pipelines\n"
                   "- infra/: Terraform and Kubernetes\n- docs/: engineering docs\n- third_party/: vendored code"),
        ("Working on one service", "```\ngit sparse-checkout set services/auth docs\n```")]),
        "CODEOWNERS": "/services/auth/ @acme-corp/identity\n/services/billing/ @acme-corp/payments\n"
                      "/web/ @acme-corp/frontend\n/ml/ @acme-corp/ml\n/infra/ @acme-corp/platform\n",
        ".gitignore": GITIGNORES["node"] + GITIGNORES["python"]}
    b.commit("Initial monorepo layout", files, by="sokafor")
    paths = []
    for svc, lang in SERVICES.items():
        if lang == "go":
            fs = {f"services/{svc}/main.go": go_file(b.rng, "main", f"runs the {svc} service.", 2),
                  f"services/{svc}/handler.go": go_file(b.rng, "main", f"handles {svc} requests.", 3)}
        elif lang == "python":
            fs = {f"services/{svc}/app.py": py_module(b.rng, f"The {svc} service.", 4),
                  f"services/{svc}/index.py": py_module(b.rng, f"{svc} indexing.", 3)}
        else:
            fs = {f"services/{svc}/index.js": js_module(b.rng, f"The {svc} service.", 4, esm=False)}
        paths += list(fs)
        b.commit(f"{svc}: initial service", fs, by=pick(b.rng, ["sokafor", "dalvarez", "omar-s"]))
    web = {f"web/src/{n}.jsx": js_module(b.rng, f"The {n} page.", 3) for n in ("Home", "Cart", "Checkout", "Account")}
    b.commit("web: initial app", web, by="grace-l")
    paths += list(web)
    ml = {"ml/pipelines/train_ranker.py": py_module(b.rng, "Train the search ranker.", 5),
          "ml/pipelines/features.py": py_module(b.rng, "Feature extraction.", 4)}
    b.commit("ml: ranker pipeline", ml, by="ravi-k")
    paths += list(ml)
    docs = {f"docs/{n}.md": md_doc(b.rng, n.replace("-", " ").title(), ["overview", "runbook", "faq"])
            for n in ("onboarding", "release-process", "on-call", "architecture")}
    b.commit("docs: engineering handbook", docs, by="ana-p")
    paths += list(docs)
    b.commit("third_party: vendor protobuf and grpc", {
        "third_party/protobuf-27.2.tar.gz": Lazy(int(24 * MB), b"\x1f\x8b\x08\x00", "protobuf"),
        "third_party/grpc-1.65.tar.gz": Lazy(int(31 * MB), b"\x1f\x8b\x08\x00", "grpc")}, by="sokafor")
    b.commit("web: add images", {f"web/public/img/{n}.jpg": jpg(int(b.rng.uniform(0.6, 1.4) * MB))
                                 for n in ("hero", "promo1", "promo2", "team", "office")}, by="grace-l")
    b.commit("infra: terraform and k8s", {"infra/terraform/main.tf": 'module "eks" {\n  source = "./modules/eks"\n}\n',
                                          "infra/k8s/deploy.yaml": "apiVersion: apps/v1\nkind: Deployment\n"},
             by="kwame-a")
    team = ["sokafor", "dalvarez", "omar-s", "grace-l", "ravi-k", "ana-p", "kwame-a", "elif-k", "jberg", "zara-q",
            "felix-w", "noor-h"]
    for week in range(36):
        for _ in range(9):
            p = pick(b.rng, paths)
            area = p.split("/")[1] if p.startswith("services/") else p.split("/")[0]
            who = pick(b.rng, team)
            old = b.read(p)
            if p.endswith(".py"):
                new = old.rstrip("\n") + "\n\n\n" + py_function(b.rng)
            elif p.endswith(".md"):
                new = old.rstrip("\n") + "\n\n" + paragraph(b.rng, None, 2) + "\n"
            elif p.endswith(".go"):
                new = old.rstrip("\n") + "\n\n" + go_file(b.rng, "x", "", 1).split(")\n\n", 1)[1]
            else:
                new = old.rstrip("\n") + "\n\n" + js_module(b.rng, "", 1).split("\n", 2)[2]
            b.commit(f"{area}: {sentence(b.rng)[:60].rstrip('.,')}", {p: new}, by=who, hours=(0, 4))
        if week % 6 == 5:
            b.commit(f"third_party: bump grpc (week {week})", {"third_party/grpc-1.65.tar.gz":
                                                               Lazy(int(31 * MB), b"\x1f\x8b\x08\x00",
                                                                    f"grpc-{week}")} if week == 17 else
                     {"third_party/VERSIONS": f"grpc 1.65.{week}\nprotobuf 27.2\n"}, by="sokafor")
            b.tag(f"release-{week // 6 + 1}", f"Release train {week // 6 + 1}", by="sokafor")
    b.repo.protected["main"] = {"pr": True, "reviews": 1}


# ============================================================================ 35. kernel-lab/tiny-kernel

SUBSYS = {"mm": ["page_alloc", "slab", "vmscan", "mmap"], "fs": ["namei", "inode", "ext2", "pipe"],
          "kernel": ["sched", "fork", "signal", "time"], "drivers/net": ["e1000", "virtio_net"],
          "drivers/tty": ["serial", "vt"], "arch/x86": ["boot", "irq", "traps"], "net": ["tcp", "udp", "ipv4"]}
MAINTAINERS = {"mm": "aiko-t", "fs": "jberg", "kernel": "kwame-a", "drivers/net": "omar-s", "drivers/tty": "ben-o",
               "arch/x86": "felix-w", "net": "elif-k"}


@recipe("kernel-lab", "tiny-kernel", "A small Unix-like teaching kernel with a long, maintainer-driven history",
        language="C", topics=["kernel", "operating-system", "c"], stars=7400, kind="medium", default_branch="master",
        license="GPL-2.0",
        about="A C kernel (~100 MB with its firmware blobs) on master with a long history: subsystem "
              "maintainers send merges ('Merge branch mm-fixes ...'), dozens of contributors, Signed-off-by "
              "trailers, and annotated tags v0.1-v1.0. Made for shortlog, log --author/--grep, describe, blame "
              "and bisect at scale.")
def _kernel(b):
    for login, (name, email, tz) in {"linh-t": ("Linh Tran", "linh@tiny-kernel.org", "+0700")}.items():
        PEOPLE.setdefault(login, (name, email, tz))
    files = {"README": "tiny-kernel: a small Unix-like kernel for teaching.\n\nBuild with `make`, run with "
                       "`make qemu`.\n",
             "Makefile": "VERSION = 0.1\nobj-y := " + " ".join(f"{d}/" for d in SUBSYS) +
                         "\n\nall:\n\t$(MAKE) -C arch/x86\n",
             "COPYING": license_text("GPL-2.0", 2024, "the tiny-kernel authors"),
             "MAINTAINERS": "".join(f"{d.upper()}\nM:\t{MAINTAINERS[d]}\nF:\t{d}/\n\n" for d in SUBSYS)}
    for d, mods in SUBSYS.items():
        for m in mods:
            files[f"{d}/{m}.c"] = c_file(b.rng, f"{d}/{m}.c", 3)
    files["include/linux/kernel.h"] = "#ifndef _LINUX_KERNEL_H\n#define _LINUX_KERNEL_H\n#define ARRAY_SIZE(x) " \
                                      "(sizeof(x) / sizeof((x)[0]))\n#endif\n"
    b.commit("Initial import", files, by="linh-t")
    b.commit("firmware: add NIC and GPU blobs", {
        "firmware/e1000/e1000e-1.2.bin": Lazy(int(18 * MB), b"\x7fFW", "fw-e1000"),
        "firmware/gpu/gpu-ucode-3.bin": Lazy(int(46 * MB), b"\x7fFW", "fw-gpu"),
        "firmware/wifi/iwl-9000.ucode": Lazy(int(14 * MB), b"\x7fFW", "fw-iwl"),
        "firmware/dsp/audio-dsp-2.fw": Lazy(int(19 * MB), b"\x7fFW", "fw-dsp")}, by="linh-t")
    b.tag("v0.1", "tiny-kernel 0.1", by="linh-t")
    devs = ["mira-chen", "tomas-ruiz", "priya-n", "dalvarez", "noor-h", "yuki-s", "ravi-k", "ana-p", "chloe-m",
            "grace-l", "zara-q", "sokafor", "lena-f"]
    version = 1
    for cycle in range(9):
        for d, maint in MAINTAINERS.items():
            br = f"{d.replace('/', '-')}-{'fixes' if cycle % 2 else 'next'}"
            b.branch(br)
            for _ in range(b.rng.randint(5, 9)):
                m = pick(b.rng, SUBSYS[d])
                p = f"{d}/{m}.c"
                who = pick(b.rng, devs + [maint])
                name = PEOPLE[who][0]
                summary = pick(b.rng, [f"fix NULL dereference in {ident(b.rng)}", f"use ARRAY_SIZE in {m}",
                                       f"simplify {ident(b.rng)}", f"handle -ENOMEM in {ident(b.rng)}",
                                       f"remove unused variable in {m}", f"add missing lock in {ident(b.rng)}",
                                       f"document {ident(b.rng)}"])
                body = f"{sentence(b.rng)}\n\nSigned-off-by: {name} <{PEOPLE[who][1]}>\n"
                if who != maint:
                    body += f"Signed-off-by: {PEOPLE[maint][0]} <{PEOPLE[maint][1]}>\n"
                b.edit(p, lambda t: t.rstrip("\n") + "\n\n" + c_file(b.rng, "", 1).split("\n\n", 1)[1],
                       f"{d.split('/')[-1]}: {summary}\n\n{body}", by=who, committer=maint, hours=(1, 20))
            b.switch("master")
            b.merge(br, f"Merge branch '{br}' of git://git.tiny-kernel.org/{maint}/{d.replace('/', '-')}\n\n"
                        f"Pull {d} updates from {PEOPLE[maint][0]}.", by="linh-t", delete=True)
        if cycle % 2 == 1:
            version += 1
            tag = f"v0.{version}" if version < 10 else "v1.0"
            b.edit("Makefile", lambda t, v=tag[1:]: "VERSION = " + v + "\n" + t.split("\n", 1)[1],
                   f"tiny-kernel {tag}\n\nSigned-off-by: Linh Tran <linh@tiny-kernel.org>", by="linh-t")
            b.tag(tag, f"tiny-kernel {tag}", by="linh-t")
    b.issue(101, "Boot hangs on QEMU 9 since v0.4", "zara-q", labels=["bug", "regression"])
