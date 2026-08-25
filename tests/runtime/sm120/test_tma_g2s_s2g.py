"""M5 vertical slice: TMA G2S + mbarrier + TMA S2G roundtrip on the 5090.

Kernel (hand-written MLIR): TMA-loads a 4x8 f32 tile via mbarrier
complete_tx protocol, plain-copies SMEM->gmem (golden path 1), then
increments and TMA-stores to a second tensor (golden path 2).
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from self_cutedsl.compiler import compile_mlir_to_ptx, entry_names  # noqa: E402
from self_cutedsl.runtime import DriverJit, LaunchManifest  # noqa: E402
from self_cutedsl.runtime.tensor_map import (  # noqa: E402
    TensorMapRecipe, CUtensorMapView)

MLIR_SRC = ROOT / "tests/ptx/tma_g2s_s2g.mlir"


@pytest.mark.sm120
def test_tma_roundtrip():
    if not torch.cuda.is_available():
        pytest.fail("SM120 release-gate: GPU required")

    ptx = compile_mlir_to_ptx(MLIR_SRC)
    for needle in ("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
                   "mbarrier.arrive.expect_tx",
                   "mbarrier.try_wait.parity",
                   "cp.async.bulk.tensor.2d.global.shared::cta"):
        assert needle in ptx, f"missing PTX instruction: {needle}"

    torch.manual_seed(0)
    src = torch.arange(32, device="cuda", dtype=torch.float32).reshape(4, 8) / 7.0
    dst = torch.zeros(4, 8, device="cuda", dtype=torch.float32)
    plain_out = torch.zeros(32, device="cuda", dtype=torch.float32)

    recipe = TensorMapRecipe(dtype=torch.float32, shape=(4, 8),
                             strides_elems=(8, 1), box=(8, 4))
    tma_in = CUtensorMapView(recipe, src)
    tma_out = CUtensorMapView(recipe, dst)

    jit = DriverJit(ptx)
    manifest = LaunchManifest(
        entry="tma_roundtrip",
        args=[{"name": "tma_in", "type": "ptr"},
              {"name": "tma_out", "type": "ptr"},
              {"name": "gmem_out", "type": "ptr"}],
        grid=(1, 1, 1),
        block=(32, 1, 1),
    )
    jit.launch(manifest, tma_in.device_copy, tma_out.device_copy, plain_out)
    jit.synchronize()
    torch.cuda.synchronize()

    # golden 1: plain copy saw the TMA-loaded tile
    torch.testing.assert_close(plain_out.reshape(4, 8), src)
    # golden 2: TMA-stored tile is the incremented tile
    torch.testing.assert_close(dst, src + 100.0)
    print("M5_TMA_OK:", torch.cuda.get_device_name(0))
