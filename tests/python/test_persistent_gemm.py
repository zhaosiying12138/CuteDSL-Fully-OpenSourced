"""M6.5: full-matrix persistent TMA GEMM with TMA S2G epilogue.

Grid-stride persistent scheduling over all output tiles (dense_gemm's
shape), 8 warps, 2-stage pipeline, per-tile barrier re-init, result
written back via cp.async.bulk.tensor S2G (no plain gmem stores).
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

TILE_M, TILE_N, TILE_K = 64, 64, 64
WARPS_M, WARPS_N = 4, 2
STAGES = 2


@cute.kernel
def persistent_gemm_kernel(mA: cute.TmaTensor, mB: cute.TmaTensor,
                           mC: cute.TmaTensor, m_ctas: cutlass.Int32,
                           k_tiles: cutlass.Int32,
                           M: cutlass.Constexpr, N: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    is0 = tidx == 0

    sA = cute.make_smem_tile("ps_sA", STAGES * TILE_M * TILE_K,
                             element=cutlass.Float16)
    sB = cute.make_smem_tile("ps_sB", STAGES * TILE_K * TILE_N,
                             element=cutlass.Float16)
    sC = cute.make_smem_tile("ps_sC", TILE_M * TILE_N,
                             element=cutlass.Float32)
    bar0 = cute.make_mbarrier("ps_bar0", 1)
    bar1 = cute.make_mbarrier("ps_bar1", 1)

    tx_ab = TILE_M * TILE_K * 2 + TILE_K * TILE_N * 2

    tiles_m = M // TILE_M
    tiles_n = N // TILE_N
    total_tiles = tiles_m * tiles_n

    wid = tidx // 32
    lane = tidx % 32
    lane16 = lane % 16
    half = lane // 16
    warp_m = wid % WARPS_M
    warp_n = wid // WARPS_M
    group = lane // 4
    tig = lane % 4
    t2 = tig * 2
    NATOMS_N = 4
    EPI_TMA = True

    for work_id in range(bidx, total_tiles, m_ctas):
        tm = work_id // tiles_n
        tn = work_id % tiles_n

        acc = [cute.zero_f32() for _ in range(4 * NATOMS_N)]

        # prologue: stage0 <- tile kt0, stage1 <- tile kt1
        if is0:
            cute.mbarrier_arrive_expect_tx(bar0, tx_ab)
            cute.tma_load(mA, cute.smem_stage(sA, 0, TILE_M * TILE_K), bar0,
                          [0, tm * TILE_M])
            cute.tma_load(mB, cute.smem_stage(sB, 0, TILE_K * TILE_N), bar0,
                          [tn * TILE_N, 0])
            cute.mbarrier_arrive_expect_tx(bar1, tx_ab)
            cute.tma_load(mA, cute.smem_stage(sA, 1, TILE_M * TILE_K), bar1,
                          [TILE_K, tm * TILE_M])
            cute.tma_load(mB, cute.smem_stage(sB, 1, TILE_K * TILE_N), bar1,
                          [tn * TILE_N, TILE_K])
        cute.sync_threads()

        for kt in range(0, k_tiles):
            stage = kt % 2
            parity = (kt // 2) % 2
            use0 = stage == 0
            sAs = cute.smem_stage(sA, stage, TILE_M * TILE_K)
            sBs = cute.smem_stage(sB, stage, TILE_K * TILE_N)
            if use0:
                cute.mbarrier_try_wait_parity(bar0, parity)
            else:
                cute.mbarrier_try_wait_parity(bar1, parity)
            cute.fence_proxy()
            cute.sync_threads()

            for kk in range(TILE_K // 16):
                fa = cute.ldmatrix(
                    sAs, (warp_m * 16 + lane16) * TILE_K + kk * 16 + half * 8,
                    num=4, trans=False)
                a_frags = [cute.extract_frag(fa, i) for i in range(4)]
                for na in range(NATOMS_N):
                    fb = cute.ldmatrix(
                        sBs, (kk * 16 + lane16) * TILE_N
                        + (warp_n * 4 + na) * 8,
                        num=2, trans=True)
                    b_frags = [cute.extract_frag(fb, i) for i in range(2)]
                    sub = cute.gemm(None, [acc[na * 4 + i] for i in range(4)],
                                    a_frags, b_frags)
                    acc = acc[:na * 4] + sub + acc[(na + 1) * 4:]

            cute.sync_threads()
            # produce: refill this stage with tile kt+2 (skip past-the-end:
            # no in-flight TMA write may race the next tile's prologue)
            kt32 = cute.idx_to_i32(kt)
            ktn = cute.add_i32(kt32, STAGES)
            do_refill = cute.lt_i32(ktn, k_tiles)
            if do_refill:
                if is0:
                    if use0:
                        cute.mbarrier_arrive_expect_tx(bar0, tx_ab)
                        cute.tma_load(
                            mA, cute.smem_stage(sA, 0, TILE_M * TILE_K),
                            bar0, [ktn * TILE_K, tm * TILE_M])
                        cute.tma_load(
                            mB, cute.smem_stage(sB, 0, TILE_K * TILE_N),
                            bar0, [tn * TILE_N, ktn * TILE_K])
                    else:
                        cute.mbarrier_arrive_expect_tx(bar1, tx_ab)
                        cute.tma_load(
                            mA, cute.smem_stage(sA, 1, TILE_M * TILE_K),
                            bar1, [ktn * TILE_K, tm * TILE_M])
                        cute.tma_load(
                            mB, cute.smem_stage(sB, 1, TILE_K * TILE_N),
                            bar1, [tn * TILE_N, ktn * TILE_K])

        # epilogue: SMEM C tile -> TMA S2G
        for na in range(NATOMS_N):
            sub = [acc[na * 4 + i] for i in range(4)]
            col = (warp_n * 4 + na) * 8 + t2
            row = warp_m * 16 + group
            sC[row * TILE_N + col] = sub[0]
            sC[row * TILE_N + col + 1] = sub[1]
            sC[(row + 8) * TILE_N + col] = sub[2]
            sC[(row + 8) * TILE_N + col + 1] = sub[3]
        cute.sync_threads()
        if EPI_TMA:
            cute.fence_proxy()
            if is0:
                cute.tma_store(mC, sC, [tn * TILE_N, tm * TILE_M])
        else:
            for na in range(NATOMS_N):
                sub2 = [acc[na * 4 + i] for i in range(4)]
                col = (warp_n * 4 + na) * 8 + t2
                row = warp_m * 16 + group
                g0 = (tm * TILE_M + row) * N + tn * TILE_N + col
                mC[g0] = sub2[0]
                mC[g0 + 1] = sub2[1]
                mC[g0 + 8 * N] = sub2[2]
                mC[g0 + 8 * N + 1] = sub2[3]
        cute.sync_threads()
        # re-init barriers so the next tile's phases start clean
        if is0:
            cute.mbarrier_reinit(bar0, 1)
            cute.mbarrier_reinit(bar1, 1)
        cute.fence_and_sync()


@cute.jit
def run_persistent_gemm(a, b, c, m_ctas: cutlass.Int32, k_tiles: cutlass.Int32,
                        M: cutlass.Constexpr, N: cutlass.Constexpr,
                        G: cutlass.Constexpr = 4):
    persistent_gemm_kernel(a, b, c, m_ctas, k_tiles, M, N).launch(
        grid=(G, 1, 1), block=(32 * WARPS_M * WARPS_N, 1, 1))


@pytest.mark.sm120
@pytest.mark.parametrize("M,N,K,G", [
    (128, 128, 128, 4),    # 4 tiles / 4 CTAs
    (128, 128, 256, 4),    # k_tiles=4 with refills
    (256, 128, 128, 8),    # 8 tiles / 8 CTAs
    (256, 256, 128, 16),   # 16 tiles / 16 CTAs
    (128, 256, 256, 8),
    (256, 128, 128, 4),    # MULTI-TILE per CTA (2 tiles)
    (256, 128, 256, 4),    # multi-tile + k_tiles=4 refills
    (256, 256, 128, 4),    # 16 tiles / 4 CTAs (4 tiles each)
    (512, 512, 512, 64),   # full occupancy wave
])
def test_persistent_tma_gemm(M, N, K, G):
    cutlass.cuda.initialize_cuda_context()
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    ra = TensorMapRecipe(dtype=torch.float16, shape=(M, K),
                         strides_elems=(K, 1), box=(TILE_K, TILE_M))
    rb = TensorMapRecipe(dtype=torch.float16, shape=(K, N),
                         strides_elems=(N, 1), box=(TILE_N, TILE_K))
    rc = TensorMapRecipe(dtype=torch.float32, shape=(M, N),
                         strides_elems=(N, 1), box=(TILE_N, TILE_M))

    run_persistent_gemm(CUtensorMapView(ra, a), CUtensorMapView(rb, b),
                        CUtensorMapView(rc, c), G, K // TILE_K, M, N, G=G)
    torch.cuda.synchronize()

    expected = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(c, expected, atol=0.5, rtol=0.5)
