"""Perf-suite smoke regression (opt-in via -m perf).

Keeps the tools/perf bench scripts honest without paying the full benchmark
cost in the correctness suite: runs each flashinfer bench on one tiny proven
shape and asserts a JSON capture with a PASS case comes out.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/perf"))

pytestmark = [pytest.mark.sm120, pytest.mark.perf]


def _run_main(module, argv, tmp_path):
    old_argv = sys.argv
    sys.argv = [module.__file__, *argv, "--out-dir", str(tmp_path)]
    try:
        rc = module.main()
    finally:
        sys.argv = old_argv
    assert rc == 0


def test_norm_bench_smoke(tmp_path):
    import bench_flashinfer_norm as B
    _run_main(B, ["--warmup", "1", "--iters", "3",
                  "--shapes", "7x256"], tmp_path)
    data = json.loads(
        (tmp_path / "norm_rmsnorm_fp4quant_self.json").read_text())
    assert data["cases"] and data["cases"][0].get("median_us"), data
    data = json.loads(
        (tmp_path / "norm_add_rmsnorm_fp4quant_self.json").read_text())
    assert data["cases"] and data["cases"][0].get("median_us"), data


def test_b12x_bench_smoke(tmp_path):
    import bench_flashinfer_b12x as B
    _run_main(B, ["--warmup", "1", "--iters", "2",
                  "--shapes", "2:256:128:1:64"], tmp_path)
    data = json.loads((tmp_path / "b12x_moe_self.json").read_text())
    assert data["cases"] and data["cases"][0].get("median_us"), data
