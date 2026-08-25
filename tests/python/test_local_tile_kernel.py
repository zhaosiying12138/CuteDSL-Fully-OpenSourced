"""M6.2: in-kernel layout algebra — dense_gemm's local_tile pattern.

Kernel-side hierarchical views with SSA rest coordinates:
  gA_mkl = cute.local_tile(mA, slice_(tile, (None, 0, None)), (None,)*3)
  tile   = gA_mkl[(None, None), (bm, bk)]   # SSA coords
then TMA GEMM consuming the selected tile. Golden vs torch.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/cutlass_compat"))

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from self_cutedsl.runtime.tensor_map import TensorMapRecipe, CUtensorMapView

TILE_M, TILE_N, TILE_K = 64, 8, 64
BLOCKS_M = 2          # grid covers 2 output tiles via bidx


@cute.kernel
def lt_gemm_kernel(mA: cute.TmaTensor, mB: cute.TmaTensor, mC: cute.Tensor,
                   k_tiles: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    is0 = tidx == 0

    sA = cute.make_smem_tile("lt_sA", TILE_M * TILE_K)
    sB = cute.make_smem_tile("lt_sB", TILE_K * TILE_N)
    bar = cute.make_mbarrier("lt_bar", 1)

    tiled_mma = cute.make_tiled_mma(
        cute.nvgpu.warp.MmaF16BF16Op(), (TILE_M // 16, TILE_N // 8, 1))

    acc = [cute.zero_f32() for _ in range(4)]

    for kt in range(0, k_tiles):
        cute.sync_threads()
        if is0:
            cute.mbarrier_arrive_expect_tx(bar,
                TILE_M * TILE_K * 2 + TILE_K * TILE_N * 2)
            cute.tma_load(mA, sA, bar, [kt * TILE_K, bidx * TILE_M])
            cute.tma_load(mB, sB, bar, [bidx * TILE_N, kt * TILE_K])
        cute.mbarrier_try_wait_parity(bar, kt % 2)
        cute.sync_threads()

        wid = tidx // 32
        lane = tidx % 32
        lane16 = lane % 16
        half = lane // 16
        for kk in range(TILE_K // 16):
            fa = cute.ldmatrix(
                sA, (wid * 16 + lane16) * TILE_K + kk * 16 + half * 8,
                num=4, trans=False)
            fb = cute.ldmatrix(sB, (kk * 16 + lane16) * TILE_N,
                               num=2, trans=True)
            a_frags = [cute.extract_frag(fa, i) for i in range(4)]
            b_frags = [cute.extract_frag(fb, i) for i in range(2)]
            acc = cute.gemm(tiled_mma, acc, a_frags, b_frags)

    # epilogue with bidx-selected output tile via local_tile-style offset
    wid = tidx // 32
    lane = tidx % 32
    group = lane // 4
    tig = lane % 4
    t2 = tig * 2
    row0 = (bidx * TILE_M + wid * 16 + group) * (TILE_N * BLOCKS_M) \
        + bidx * TILE_N + t2
    mC[row0] = acc[0]
    mC[row0 + 1] = acc[1]
    mC[row0 + 8 * TILE_N * BLOCKS_M] = acc[2]
    mC[row0 + 8 * TILE_N * BLOCKS_M + 1] = acc[3]


@cute.jit
def run_lt_gemm(a, b, c, k_tiles: cutlass.Int32):
    lt_gemm_kernel(a, b, c, k_tiles).launch(
        grid=(BLOCKS_M, 1, 1), block=(32 * (TILE_M // 16) * (TILE_N // 8), 1, 1))


@pytest.mark.sm120
@pytest.mark.parametrize("K", [64, 128])
def test_local_tile_tma_gemm(K):
    cutlass.cuda.initialize_cuda_context()
    M = TILE_M * BLOCKS_M
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, TILE_N * BLOCKS_M, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, TILE_N * BLOCKS_M, device="cuda", dtype=torch.float32)

    recipe_a = TensorMapRecipe(dtype=torch.float16, shape=(M, K),
                               strides_elems=(K, 1), box=(TILE_K, TILE_M))
    recipe_b = TensorMapRecipe(dtype=torch.float16, shape=(K, TILE_N * BLOCKS_M),
                               strides_elems=(TILE_N * BLOCKS_M, 1),
                               box=(TILE_N, TILE_K))

    run_lt_gemm(CUtensorMapView(recipe_a, a), CUtensorMapView(recipe_b, b),
                c, K // TILE_K)
    torch.cuda.synchronize()

    # each CTA computes its own (TILE_M x TILE_N) tile on the block
    # diagonal: rows [b*TM, (b+1)*TM) x cols [b*TN, (b+1)*TN)
    full = torch.matmul(a.float(), b.float())
    expected = torch.zeros_like(full)
    for bidx in range(BLOCKS_M):
        expected[bidx * TILE_M:(bidx + 1) * TILE_M,
                 bidx * TILE_N:(bidx + 1) * TILE_N] = \
            full[bidx * TILE_M:(bidx + 1) * TILE_M,
                 bidx * TILE_N:(bidx + 1) * TILE_N]
    torch.testing.assert_close(c, expected, atol=0.5, rtol=0.5)
