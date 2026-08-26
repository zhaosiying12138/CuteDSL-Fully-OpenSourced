"""CPU regression for the block-scaled shared-memory stage calculation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/cutlass_compat"))

import cutlass
import cutlass.cute as cute
from cutlass.utils import blockscaled_layout


KERNEL_DIR = (
    ROOT
    / "third_party/cutlass/examples/python/CuTeDSL/cute/blackwell_geforce/kernel"
    / "blockscaled_gemm"
)
sys.path.insert(0, str(KERNEL_DIR))

from dense_blockscaled_gemm_persistent_cooperative import (  # noqa: E402
    Sm120BlockScaledGemmKernel,
)


def test_default_nvfp4_stage_count_stays_bounded():
    tile = (128, 128, 128)
    sf_vec_size = 16
    sfa = blockscaled_layout.sm120_make_smem_layout_sfa(
        object(), tile, sf_vec_size, 1
    )
    sfb = blockscaled_layout.sm120_make_smem_layout_sfb(
        object(), tile, sf_vec_size, 1
    )

    ab_stage, epi_stage = Sm120BlockScaledGemmKernel._compute_stages(
        tile,
        cutlass.Float4E2M1FN,
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        sfa,
        sfb,
        (128, 128),
        cutlass.Float16,
        (100 - 1) * 1024,
        1,
    )

    assert cute.size((128, 128)) == 16_384
    assert cute.size(cute.filter_zeros(sfa).shape) == 1_024
    assert cute.size(cute.filter_zeros(sfb).shape) == 1_024
    assert (ab_stage, epi_stage) == (3, 1)
