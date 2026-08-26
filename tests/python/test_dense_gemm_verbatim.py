"""S5: the official blackwell_geforce dense_gemm.py running VERBATIM on the
self-hosted stack (frontend trace -> cute/MLIR -> cutlass-compiler passes ->
PTX -> Driver JIT -> 5090), golden-compared against torch einsum.

The kernel source is imported unmodified from third_party/cutlass examples.
Tile (64,64,64) is the configuration the example ships with; larger tiles
exceed SMEM (tile_m=128, skipped by the official runner too) or need the
multi-epi-tile epilogue (tile_n=128) — recorded as boundaries in README.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/cutlass_compat"))
sys.path.insert(0, str(ROOT / "third_party/cutlass/examples/python/CuTeDSL"))
sys.path.insert(
    0,
    str(ROOT / "third_party/cutlass/examples/python/CuTeDSL/cute/"
               "blackwell_geforce/kernel/dense_gemm"))

import cutlass
import dense_gemm


@pytest.mark.sm120
@pytest.mark.parametrize("mnkl", [
    (128, 128, 128, 1),   # example default: official tolerance passes
    (128, 256, 128, 1),
    (256, 256, 128, 1),
    (128, 128, 256, 1),   # 4 k-tiles: double stage wrap through the pipeline
])
def test_dense_gemm_verbatim(mnkl):
    cutlass.cuda.initialize_cuda_context()
    torch.manual_seed(0)
    dense_gemm.run(
        mnkl, cutlass.Float16, cutlass.Float16, cutlass.Float16,
        cutlass.Float32, "k", "k", "n",
        tile_shape_mnk=(64, 64, 64), tolerance=0.05,
        warmup_iterations=0, iterations=1, skip_ref_check=False)
