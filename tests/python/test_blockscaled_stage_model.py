"""Block-scaled host-layout, conversion, and SM120 golden regressions."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/cutlass_compat"))

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import blockscaled_layout
from self_cutedsl.frontend.cute_objects import _flatten


KERNEL_DIR = (
    ROOT
    / "third_party/cutlass/examples/python/CuTeDSL/cute/blackwell_geforce/kernel"
    / "blockscaled_gemm"
)
sys.path.insert(0, str(KERNEL_DIR))

from dense_blockscaled_gemm_persistent_cooperative import (  # noqa: E402
    Sm120BlockScaledGemmKernel,
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L,
    run_bs,
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


def test_sf_layout_is_derived_from_logical_ab_k_extent():
    layout = blockscaled_layout.tile_atom_to_shape_SF((128, 128, 1), 16)

    assert tuple(_flatten(layout.shape)) == (32, 4, 1, 4, 2, 1)
    assert tuple(_flatten(layout.stride)) == (16, 4, 1024, 1, 512, 1024)


def test_host_only_sf_scatter_runs_for_sfa_and_same_shaped_sfb():
    src_a = torch.arange(128 * 8, dtype=torch.float32).reshape(128, 8, 1)
    src_b = src_a + 4096
    base_a = torch.zeros((1, 1, 2, 32, 4, 4), dtype=torch.float32)
    base_b = torch.zeros_like(base_a)
    dst_a = base_a.permute(3, 4, 1, 5, 2, 0)
    dst_b = base_b.permute(3, 4, 1, 5, 2, 0)

    cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        from_dlpack(src_a), from_dlpack(dst_a)
    )
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        from_dlpack(src_b), from_dlpack(dst_b)
    )

    for m in range(128):
        for kg in range(8):
            coord = (m % 32, m // 32, 0, kg % 4, kg // 4, 0)
            assert dst_a[coord] == src_a[m, kg, 0]
            assert dst_b[coord] == src_b[m, kg, 0]


def test_random_initializer_uses_integer_values_and_exclusive_max():
    torch.manual_seed(0)
    values = cutlass_torch.create_and_permute_torch_tensor(
        (16, 16),
        torch.float32,
        permute_order=(0, 1),
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(min_val=1, max_val=3),
    )

    assert set(values.unique().tolist()) == {1.0, 2.0}


def test_matrix_major_changes_strides_not_logical_shape():
    k_major = cutlass_torch.matrix(2, 17, 34, False, cutlass.Float32)
    m_major = cutlass_torch.matrix(2, 17, 34, True, cutlass.Float32)

    assert k_major.shape == m_major.shape == (17, 34, 2)
    assert k_major.stride()[1] == 1
    assert m_major.stride()[0] == 1
    assert k_major.stride() != m_major.stride()


@pytest.mark.sm120
@pytest.mark.parametrize("mnkl", [
    (128, 128, 128, 1),
    (128, 128, 256, 1),  # second global K tile exercises stage-1 SF window
])
def test_nvfp4_random_scale_factors_match_torch_golden(mnkl):
    """Full official cooperative path with every K16 SF group varying."""
    cutlass.cuda.initialize_cuda_context()
    torch.manual_seed(0)
    run_bs(
        mnkl,
        cutlass.Float4E2M1FN,
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        16,
        cutlass.Float16,
        cutlass.Float32,
        "k",
        "k",
        "n",
        (128, 128, 128),
        (128, 128),
        0.1,
        0,
        1,
        False,
        False,
    )
