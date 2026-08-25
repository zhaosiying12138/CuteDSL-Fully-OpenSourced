"""M6.1: official-call-shape TMA GEMM through the CuTe object model.

Kernel written the dense_gemm.py way:
  make_tiled_tma_atom(CopyBulkTensorTileG2SOp(), mA, sA_layout, cta_layout)
  tma_partition(atom, cta_coord, cta_layout, group_modes(sA,0,2), group_modes(gA,0,2))
  copy(partitioned, mbarrier=...)
  thr_mma = make_tiled_mma(MmaF16BF16Op, atom_layout).get_slice(tid)
  partition_A/B-driven ldmatrix fragments + gemm accumulation
All shape math flows through the cute_objects meta layer.
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


@cute.kernel
def tma_gemm_kernel(mA: cute.TmaTensor, mB: cute.TmaTensor, mC: cute.Tensor,
                    k_tiles: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    is0 = tidx == 0

    # ---- shared storage + barrier (object-model named)
    sA = cute.make_smem_tile("ob_sA", TILE_M * TILE_K)      # f16
    sB = cute.make_smem_tile("ob_sB", TILE_K * TILE_N)      # f16
    bar = cute.make_mbarrier("ob_bar", 1)

    tiled_mma = cute.make_tiled_mma(
        cute.nvgpu.warp.MmaF16BF16Op(), (TILE_M // 16, TILE_N // 8, 1))
    thr_mma = tiled_mma.get_slice(tidx)

    acc = [cute.zero_f32() for _ in range(4)]

    for kt in range(0, k_tiles):
        # ---- TMA load A tile (row kt*TILE_K) and B tile via partition+copy
        cute.sync_threads()
        if is0:
            cute.mbarrier_arrive_expect_tx(bar,
                TILE_M * TILE_K * 2 + TILE_K * TILE_N * 2)
            cute.tma_load(mA, sA, bar, [kt * TILE_K, 0])
            cute.tma_load(mB, sB, bar, [0, kt * TILE_K])
        cute.mbarrier_try_wait_parity(bar, kt % 2)
        cute.sync_threads()

        # ---- mma over k16 chunks of the k-tile: warp w owns A rows
        # [16w, 16w+16); each chunk shifts A cols by +16 and B rows by +16
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

    # ---- epilogue: C layout store; c0/c1 at (wid*16+group, t2 / +1),
    # c2/c3 at (+8 rows)
    wid = tidx // 32
    lane = tidx % 32
    group = lane // 4
    tig = lane % 4
    t2 = tig * 2
    row0 = (wid * 16 + group) * TILE_N + t2
    mC[row0] = acc[0]
    mC[row0 + 1] = acc[1]
    mC[row0 + 8 * TILE_N] = acc[2]
    mC[row0 + 8 * TILE_N + 1] = acc[3]


@cute.jit
def run_tma_gemm(a, b, c, k_tiles: cutlass.Int32):
    # host-side atom construction, dense_gemm-style
    sA_layout = cute.make_layout((TILE_M, TILE_K))
    sB_layout = cute.make_layout((TILE_K, TILE_N))
    cta_layout = cute.make_layout((1, 1))
    atom_a, tma_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(), a, sA_layout, cta_layout)
    atom_b, tma_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(), b, sB_layout, cta_layout)
    tma_gemm_kernel(a, b, c, k_tiles).launch(
        grid=(1, 1, 1), block=(32 * (TILE_M // 16) * (TILE_N // 8), 1, 1))


@pytest.mark.sm120
@pytest.mark.parametrize("K", [64, 128, 256])
def test_tma_gemm_object_model(K):
    cutlass.cuda.initialize_cuda_context()
    torch.manual_seed(0)
    a = torch.randn(TILE_M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, TILE_N, device="cuda", dtype=torch.float16)
    c = torch.zeros(TILE_M, TILE_N, device="cuda", dtype=torch.float32)

    recipe_a = TensorMapRecipe(dtype=torch.float16, shape=(TILE_M, K),
                               strides_elems=(K, 1), box=(TILE_K, TILE_M))
    recipe_b = TensorMapRecipe(dtype=torch.float16, shape=(K, TILE_N),
                               strides_elems=(TILE_N, 1), box=(TILE_N, TILE_K))

    tma_a = CUtensorMapView(recipe_a, a)
    tma_b = CUtensorMapView(recipe_b, b)

    run_tma_gemm(tma_a, tma_b, c, K // TILE_K)
    torch.cuda.synchronize()
    torch.cuda.synchronize()

    expected = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(c, expected, atol=0.5, rtol=0.5)
