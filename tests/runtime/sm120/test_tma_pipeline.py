"""M5: 2-stage TMA pipeline with phase-parity rollover, golden check.

Tile k = the 4x8 tensor windowed at row k (rows >= 4 OOB zero-fill);
the pipeline consumes stage k%2 waiting on parity floor(k/2)%2 and
refills it with tile k+2. Result accumulates every tile's elements.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from self_cutedsl.compiler import compile_mlir_to_ptx  # noqa: E402
from self_cutedsl.runtime import DriverJit, LaunchManifest  # noqa: E402
from self_cutedsl.runtime.tensor_map import TensorMapRecipe, CUtensorMapView  # noqa: E402

MLIR_SRC = ROOT / "tests/ptx/tma_pipeline_multistage.mlir"


@pytest.mark.sm120
@pytest.mark.parametrize("n_tiles", [2, 4, 7])  # 7: multiple parity rollovers + OOB tiles
def test_tma_pipeline_multistage(n_tiles):
    if not torch.cuda.is_available():
        pytest.fail("SM120 release-gate: GPU required")

    ptx = compile_mlir_to_ptx(MLIR_SRC)
    assert ptx.count("cp.async.bulk.tensor.2d.shared") >= 4
    assert "mbarrier.try_wait.parity" in ptx

    torch.manual_seed(0)
    src = (torch.arange(32, device="cuda", dtype=torch.float32) * 3 + 1).reshape(4, 8)
    result = torch.zeros(32, device="cuda", dtype=torch.float32)

    recipe = TensorMapRecipe(dtype=torch.float32, shape=(4, 8),
                             strides_elems=(8, 1), box=(8, 4))
    tma = CUtensorMapView(recipe, src)

    jit = DriverJit(ptx)
    manifest = LaunchManifest(
        entry="tma_pipe",
        args=[{"name": "tma", "type": "ptr"},
              {"name": "result", "type": "ptr"},
              {"name": "n_tiles", "type": "i32"}],
        grid=(1, 1, 1),
        block=(32, 1, 1),
    )
    jit.launch(manifest, tma.device_copy, result, n_tiles)
    jit.synchronize()
    torch.cuda.synchronize()

    # expected: sum over k of tile_k where tile_k[r, c] = src[r+k, c] if r+k<4 else 0
    expected = torch.zeros(32, device="cuda")
    for k in range(n_tiles):
        for r in range(4):
            if r + k < 4:
                expected[r * 8:(r + 1) * 8] += src[r + k]
    torch.testing.assert_close(result, expected)
    print(f"M5_PIPELINE_OK n_tiles={n_tiles}")
