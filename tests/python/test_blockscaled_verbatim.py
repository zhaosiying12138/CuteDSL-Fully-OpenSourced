"""M7: the official blackwell_geforce blockscaled (NVFP4) cooperative kernel
running VERBATIM on the self-hosted stack with golden comparison.

Source: third_party/.../blockscaled_gemm/dense_blockscaled_gemm_persistent_
cooperative.py, imported unmodified. The sv=32 (MXFP4/e8m0) config and the
unaligned boundary shape are honest boundaries — see README.
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
    0, str(ROOT / "third_party/cutlass/examples/python/CuTeDSL/cute/"
               "blackwell_geforce/kernel/blockscaled_gemm"))

import cutlass
import dense_blockscaled_gemm_persistent_cooperative as bs


@pytest.mark.sm120
@pytest.mark.parametrize("mnkl", [
    (128, 128, 128, 1),        # minimal smoke
    (1024, 1024, 1024, 1),     # M0 baseline default (nvfp4)
    (4096, 4096, 4096, 1),     # the 411-TF/s perf shape
])
def test_blockscaled_cooperative_verbatim(mnkl):
    cutlass.cuda.initialize_cuda_context()
    torch.manual_seed(0)
    bs.run(
        mnkl,
        cutlass.Float4E2M1FN, cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN, 16, cutlass.Float16,
        a_major="k", b_major="k", c_major="n",
        tile_shape_mnk=(128, 128, 128), epi_tile=(128, 128),
        tolerance=0.1, warmup_iterations=0, iterations=1,
        skip_ref_check=False)
