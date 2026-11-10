"""COMP_GUIDE §12 Docker (generator/sections/s12_docker) on docker-lite: every template's reference session passes
in the live terminal and an idle one doesn't, every module has two templates per tier, and docker's feature names
stop at the image (`docker run IMG cmd -x` is not `docker run -x`)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from docker_templates import check, problems  # noqa: E402

PROBLEMS = [(f"{tpl.module}-{tpl.name}-{item}-{seed}", p) for tpl, item, seed, p in problems(None, 1)]


@pytest.mark.parametrize("pid,p", PROBLEMS, ids=[x[0] for x in PROBLEMS])
def test_template(pid, p):
    assert check(p) == []


def test_modules_complete():
    from generator.core import TIERS, is_live, modules, templates
    assert modules("docker") == [f"12.{i:02d}" for i in range(1, 15)]
    for m in modules("docker"):
        assert is_live("docker", m), m
        for tier in TIERS:
            assert len(templates("docker", m, tier)) >= 2, (m, tier)


def test_run_features_stop_at_image():
    from generator.bashlite.features import features
    f = features("docker run --rm --gpus all -v data:/d --user 1000 nvidia/cuda nvidia-smi -L")
    assert {"docker run --rm", "docker run --gpus", "docker run -v", "docker run --user"} <= f
    assert "docker run -L" not in f
    f = features("docker exec -it -u root api sh -c 'id -u'")
    assert {"docker exec -i", "docker exec -t", "docker exec -u"} <= f and "docker exec -c" not in f
    assert "docker compose run --rm" in features("docker compose run --rm app python -c 1")
