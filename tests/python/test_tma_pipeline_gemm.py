"""M6.3: 2-stage pipelined TMA GEMM — the dense_gemm mainloop shape.

Producer/consumer within one CTA: TMA loads k-tile t+2 into stage
(t mod 2) while compute consumes stage (t mod 2) waiting on parity
floor(t/2) mod 2 — exactly the PipelineTmaAsync protocol used by the
flagship kernel, expressed with our verified primitives.
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
from self_cutedsl.runtime.tensor_map import TensorMapRecipe, CUtensorMapView

TILE_M, TILE_N, TILE_K = 64, 8, 64
STAGES = 2
BLOCKS_M = 2


@cute.kernel
def pipe_gemm_kernel(mA: cute.TmaTensor, mB: cute.TmaTensor,
                     mC: cute.Tensor, k_tiles: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    is0 = tidx == 0

    # staged (double-buffered) smem: stage folded into element offset
    sA = cute.make_smem_tile("pg_sA", STAGES * TILE_M * TILE_K, element=cutlass.Float16)
    sB = cute.make_smem_tile("pg_sB", STAGES * TILE_K * TILE_N, element=cutlass.Float16)
    bar0 = cute.make_mbarrier("pg_bar0", 1)
    bar1 = cute.make_mbarrier("pg_bar1", 1)

    tiled_mma = cute.make_tiled_mma(
        cute.nvgpu.warp.MmaF16BF16Op(), (TILE_M // 16, TILE_N // 8, 1))

    tx = TILE_M * TILE_K * 2 + TILE_K * TILE_N * 2

    # ---- prologue: stage 0 <- tile 0, stage 1 <- tile 1 (stage-offset
    # windows into the staged arrays)
    if is0:
        cute.mbarrier_arrive_expect_tx(bar0, tx)
        cute.tma_load(mA, cute.smem_stage(sA, 0, TILE_M * TILE_K), bar0,
                      [0, bidx * TILE_M])
        cute.tma_load(mB, cute.smem_stage(sB, 0, TILE_K * TILE_N), bar0,
                      [bidx * TILE_N, 0])
        cute.mbarrier_arrive_expect_tx(bar1, tx)
        cute.tma_load(mA, cute.smem_stage(sA, 1, TILE_M * TILE_K), bar1,
                      [TILE_K, bidx * TILE_M])
        cute.tma_load(mB, cute.smem_stage(sB, 1, TILE_K * TILE_N), bar1,
                      [bidx * TILE_N, TILE_K])

    acc = [cute.zero_f32() for _ in range(4)]

    for kt in range(0, k_tiles):
        stage = kt % 2
        parity = (kt // 2) % 2
        use0 = stage == 0

        # stage-select via predicated waits + SSA stage offsets
        if use0:
            cute.mbarrier_try_wait_parity(bar0, parity)
        else:
            cute.mbarrier_try_wait_parity(bar1, parity)
        sAs = cute.smem_stage(sA, stage, TILE_M * TILE_K)
        sBs = cute.smem_stage(sB, stage, TILE_K * TILE_N)
        cute.sync_threads()

        wid = tidx // 32
        lane = tidx % 32
        lane16 = lane % 16
        half = lane // 16
        for kk in range(TILE_K // 16):
            fa = cute.ldmatrix(
                sAs, (wid * 16 + lane16) * TILE_K + kk * 16 + half * 8,
                num=4, trans=False)
            fb = cute.ldmatrix(sBs, (kk * 16 + lane16) * TILE_N,
                               num=2, trans=True)
            a_frags = [cute.extract_frag(fa, i) for i in range(4)]
            b_frags = [cute.extract_frag(fb, i) for i in range(2)]
            acc = cute.gemm(tiled_mma, acc, a_frags, b_frags)

        cute.sync_threads()
        # produce: refill this stage with tile kt+2
        if is0:
            ktn = kt + STAGES
            if use0:
                cute.mbarrier_arrive_expect_tx(bar0, tx)
                cute.tma_load(mA, cute.smem_stage(sA, 0, TILE_M * TILE_K),
                              bar0, [ktn * TILE_K, bidx * TILE_M])
                cute.tma_load(mB, cute.smem_stage(sB, 0, TILE_K * TILE_N),
                              bar0, [bidx * TILE_N, ktn * TILE_K])
            else:
                cute.mbarrier_arrive_expect_tx(bar1, tx)
                cute.tma_load(mA, cute.smem_stage(sA, 1, TILE_M * TILE_K),
                              bar1, [ktn * TILE_K, bidx * TILE_M])
                cute.tma_load(mB, cute.smem_stage(sB, 1, TILE_K * TILE_N),
                              bar1, [bidx * TILE_N, ktn * TILE_K])

    # epilogue
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
def run_pipe_gemm(a, b, c, k_tiles: cutlass.Int32):
    pipe_gemm_kernel(a, b, c, k_tiles).launch(
        grid=(BLOCKS_M, 1, 1), block=(32 * (TILE_M // 16) * (TILE_N // 8), 1, 1))


@pytest.mark.sm120
@pytest.mark.parametrize("K", [64, 128, 256])  # 4 & 8 k-tiles: rollovers
def test_pipelined_tma_gemm(K):
    cutlass.cuda.initialize_cuda_context()
    M = TILE_M * BLOCKS_M
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, TILE_N * BLOCKS_M, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, TILE_N * BLOCKS_M, device="cuda", dtype=torch.float32)

    ra = TensorMapRecipe(dtype=torch.float16, shape=(M, K),
                         strides_elems=(K, 1), box=(TILE_K, TILE_M))
    rb = TensorMapRecipe(dtype=torch.float16, shape=(K, TILE_N * BLOCKS_M),
                         strides_elems=(TILE_N * BLOCKS_M, 1),
                         box=(TILE_N, TILE_K))

    run_pipe_gemm(CUtensorMapView(ra, a), CUtensorMapView(rb, b), c,
                  K // TILE_K)
    torch.cuda.synchronize()

    full = torch.matmul(a.float(), b.float())
    expected = torch.zeros_like(full)
    for bidx in range(BLOCKS_M):
        expected[bidx * TILE_M:(bidx + 1) * TILE_M,
                 bidx * TILE_N:(bidx + 1) * TILE_N] = \
            full[bidx * TILE_M:(bidx + 1) * TILE_M,
                 bidx * TILE_N:(bidx + 1) * TILE_N]
    torch.testing.assert_close(c, expected, atol=0.5, rtol=0.5)
