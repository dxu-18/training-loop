"""12.11 GPUs & ML images."""

from ._common import P, catalog, docker, dockerfile, files, heredoc, setup, template

M = "12.11"
CUDA = "nvidia/cuda:12.4.1-base-ubuntu22.04"
TORCH = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
TRAIN = ("import argparse\nimport os\nimport torch\np = argparse.ArgumentParser()\np.add_argument('--epochs', "
         "type=int, default=1)\na = p.parse_args()\nprint('cuda', torch.cuda.is_available(), torch.cuda.device_count())"
         "\nos.makedirs('outputs', exist_ok=True)\nopen('outputs/ckpt.pt', 'w').write('weights')\nprint('saved "
         "outputs/ckpt.pt after', a.epochs, 'epochs')\n")


def _gpu_setup(n=4, *extra):
    return setup(docker(f"pull={CUDA},{TORCH}", *extra), f"#: gpus {n}")


@template(M, "lite")
def verify_gpus(rng):
    items = [
        ("docker run --gpus", f"Check GPU access from a container: run `nvidia-smi` in `{CUDA}` with all GPUs (removed "
                              f"afterwards).", f"docker run --rm --gpus all {CUDA} nvidia-smi"),
        ("docker run --gpus", f"In `{TORCH}` with all GPUs, print `torch.cuda.is_available()`.",
         f"docker run --rm --gpus all {TORCH} python -c \"import torch; print(torch.cuda.is_available())\""),
        ("docker run --gpus", f"List the GPUs (`nvidia-smi -L`) that `{CUDA}` sees with all GPUs.",
         f"docker run --rm --gpus all {CUDA} nvidia-smi -L"),
    ]
    return catalog(rng, items, _gpu_setup())


@template(M, "lite")
def without_gpus(rng):
    items = [
        ("docker run", f"See what happens without `--gpus`: print `torch.cuda.is_available()` in `{TORCH}` with no "
                       f"GPU flag.", f"docker run --rm {TORCH} python -c \"import torch; print(torch.cuda.is_available())\"",
         {"require": ["docker run"]}),
        ("docker run --gpus", f"Print `torch.cuda.device_count()` in `{TORCH}` given exactly 2 GPUs (`--gpus 2`).",
         f"docker run --rm --gpus 2 {TORCH} python -c \"import torch; print(torch.cuda.device_count())\""),
    ]
    return catalog(rng, items, _gpu_setup())


@template(M, "easy")
def pick_devices(rng):
    a, b = sorted(rng.sample(range(4), 2))
    items = [
        ("docker run --gpus", f"Give a `{CUDA}` container only GPU {a} and list what it sees (`nvidia-smi -L`).",
         f"docker run --rm --gpus '\"device={a}\"' {CUDA} nvidia-smi -L"),
        ("docker run --gpus", f"Give a `{TORCH}` container GPUs {a} and {b} and print `torch.cuda.device_count()`.",
         f"docker run --rm --gpus '\"device={a},{b}\"' {TORCH} python -c \"import torch; "
         f"print(torch.cuda.device_count())\""),
    ]
    return catalog(rng, items, _gpu_setup())


@template(M, "easy")
def gpu_inspect(rng):
    s = setup(_gpu_setup(), f"docker run -dq --name trainer --gpus all {TORCH} sleep infinity >/dev/null")
    items = [
        ("docker inspect", "Print `trainer`'s GPU device requests (`.HostConfig.DeviceRequests`) as JSON.",
         "docker inspect -f '{{json .HostConfig.DeviceRequests}}' trainer"),
        ("docker exec", "List the GPUs inside the running `trainer` container (`nvidia-smi -L` through exec).",
         "docker exec trainer nvidia-smi -L"),
    ]
    return catalog(rng, items, s)


@template(M, "medium")
def gpu_training_image(rng):
    epochs = rng.randint(2, 6)
    s = setup(_gpu_setup(), "#: cwd ~/train", files({"train.py": TRAIN, "requirements.txt": "numpy\n"}))
    df = dockerfile(f"FROM {TORCH}", "WORKDIR /workspace", "COPY requirements.txt .",
                    "RUN pip install --no-cache-dir -r requirements.txt", "COPY . .", "ENV HF_HOME=/cache/hf",
                    f'CMD ["python", "train.py", "--epochs", "{epochs}"]')
    return P(f"Containerize the training script: a Dockerfile from `{TORCH}`, working in `/workspace`, installing "
             f"`requirements.txt` (no pip cache) before copying the code, `HF_HOME=/cache/hf`, and the command `python "
             f"train.py --epochs {epochs}`. Build it as `trainer`, then run it on all GPUs with 16 GB of shared "
             f"memory, the `hf-cache` volume at `/cache/hf`, and `./outputs` mounted at `/workspace/outputs` "
             f"(removed afterwards).", s,
             heredoc("Dockerfile", df) + "\ndocker build -t trainer .\ndocker run --rm --gpus all --shm-size=16g -v "
                                         'hf-cache:/cache/hf -v "$(pwd)/outputs":/workspace/outputs trainer',
             check=["image=trainer:latest", "volumes"], post="ls outputs", require=["docker build", "docker run --gpus"])


@template(M, "medium")
def shm(rng):
    s = setup(docker("pull=mlgym/trainer:1.0"), "#: gpus 2")
    items = [
        ("docker run --shm-size", "`docker run --rm --gpus all mlgym/trainer:1.0` crashes with a *bus error* in its "
                                  "DataLoader workers: the default 64 MB `/dev/shm` is too small. Run it with 8 GB of "
                                  "shared memory.", "docker run --rm --gpus all --shm-size=8g mlgym/trainer:1.0"),
        ("docker run --ipc", "`mlgym/trainer:1.0` crashes with a bus error in its DataLoader workers. Run it on all "
                             "GPUs sharing the host's IPC namespace (`--ipc=host`) instead of raising `--shm-size`.",
         "docker run --rm --gpus all --ipc=host mlgym/trainer:1.0"),
    ]
    return catalog(rng, items, s)


@template(M, "hard")
def cuda_mismatch(rng):
    s = _gpu_setup(2, "cuda=12.4")
    return P("`docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi` fails: the image needs a "
             "newer driver than this machine has (*unsatisfied condition: cuda>=12.6*). Check what the driver "
             "supports (`nvidia-smi` on the host shows the CUDA version) and run `nvidia-smi` from the CUDA 12.4.1 "
             "base image instead.", s, f"nvidia-smi\ndocker run --rm --gpus all {CUDA} nvidia-smi",
             require=["docker run --gpus"])


@template(M, "hard")
def toolkit(rng):
    s = setup(docker(f"pull={CUDA}", "notoolkit"), "#: gpus 2")
    return P("`docker run --gpus all ...` fails with *could not select device driver \"\" with capabilities: "
             "[[gpu]]*: the NVIDIA Container Toolkit is missing. Install `nvidia-container-toolkit` (apt, sudo), "
             "register its runtime with Docker (`sudo nvidia-ctk runtime configure --runtime=docker`), restart the "
             f"docker service, and list the GPUs from `{CUDA}`.", s,
             "sudo apt install -y nvidia-container-toolkit\nsudo nvidia-ctk runtime configure --runtime=docker\n"
             f"sudo systemctl restart docker\ndocker run --rm --gpus all {CUDA} nvidia-smi -L",
             post="docker info --format '{{json .Runtimes}}'", require=["nvidia-ctk", "systemctl"])


@template(M, "challenge")
def torchrun(rng):
    s = setup(_gpu_setup(), "#: cwd ~/ddp", files({"train.py": TRAIN}))
    df = dockerfile(f"FROM {TORCH}", "WORKDIR /workspace", "COPY . .", "ENV HF_HOME=/cache/hf",
                    'CMD ["torchrun", "--standalone", "--nproc_per_node=gpu", "train.py"]')
    return P("**The guide's training image.** Write the Dockerfile from the guide: base `pytorch/pytorch:2.5.1-cuda12.4"
             "-cudnn9-runtime`, work in `/workspace`, copy the code, `HF_HOME=/cache/hf`, and launch with `torchrun "
             "--standalone --nproc_per_node=gpu train.py` (exec form). Build `ddp`, run it detached as `ddp-run` on "
             "all GPUs with `--shm-size=16g` and the `hf-cache` volume at `/cache/hf`.", s,
             heredoc("Dockerfile", df) + "\ndocker build -t ddp .\ndocker run -d --name ddp-run --gpus all "
                                         "--shm-size=16g -v hf-cache:/cache/hf ddp",
             check=["image=ddp:latest", "containers"], require=["docker build", "docker run --gpus"])


@template(M, "challenge")
def gpu_checkpoints(rng):
    a = rng.randrange(4)
    s = setup(_gpu_setup(), "#: cwd ~/exp", files({"train.py": TRAIN}))
    return P(f"**Persist the checkpoints.** Run `train.py` from this folder in `{TORCH}` on GPU {a} only: mount the "
             f"folder at `/workspace`, work there, run `python train.py --epochs 3` with 8 GB of shared memory, and run "
             f"as **your** user so `outputs/ckpt.pt` on the host belongs to you (removed afterwards).", s,
             f"docker run --rm --gpus '\"device={a}\"' --shm-size=8g --user \"$(id -u):$(id -g)\" -v \"$(pwd)\":"
             f"/workspace -w /workspace {TORCH} python train.py --epochs 3",
             post="stat -c '%U %n' outputs/ckpt.pt", require=["docker run --gpus", "docker run --user|docker run -u"])
