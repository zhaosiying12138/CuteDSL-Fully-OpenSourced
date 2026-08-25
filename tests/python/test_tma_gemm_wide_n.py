"""M6.4: wide-N pipelined TMA GEMM — 8 warps, N-direction atom tiling.

Tile 64x64: 4 warps in M (m16 each) x 2 warps in N (n8 x 8 atom
repetitions per warp = n64). B tile is (K x 64); each N-warp reads its
own 8-column window of SMEM B. Epilogue writes the full 64-wide C tile.
2-stage pipeline as in M6.3.
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
NW = WARPS_M * WARPS_N          # 8 warps


@cute.kernel
def wide_n_gemm_kernel(mA: cute.TmaTensor, mB: cute.TmaTensor,
                       mC: cute.Tensor, k_tiles: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    is0 = tidx == 0

    sA = cute.make_smem_tile("wn_sA", STAGES * TILE_M * TILE_K,
                             element=cutlass.Float16)
    sB = cute.make_smem_tile("wn_sB", STAGES * TILE_K * TILE_N,
                             element=cutlass.Float16)
    bar0 = cute.make_mbarrier("wn_bar0", 1)
    bar1 = cute.make_mbarrier("wn_bar1", 1)

    tx = TILE_M * TILE_K * 2 + TILE_K * TILE_N * 2

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

    # this thread's warp position
    wid = tidx // 32
    lane = tidx % 32
    warp_m = wid % WARPS_M               # M direction warp
    warp_n = wid // WARPS_M              # N direction warp
    lane16 = lane % 16
    half = lane // 16
    group = lane // 4
    tig = lane % 4
    t2 = tig * 2

    NATOMS_N = 4                                 # 2 warps x 4 n8 atoms = 64 cols
    acc = [cute.zero_f32() for _ in range(4 * NATOMS_N)]

    for kt in range(0, k_tiles):
        stage = kt % 2
        parity = (kt // 2) % 2
        use0 = stage == 0
        # stage offsets as SSA arithmetic BEFORE the predicated wait
        # (values defined inside scf.if regions cannot escape)
        sAs = cute.smem_stage(sA, stage, TILE_M * TILE_K)
        sBs = cute.smem_stage(sB, stage, TILE_K * TILE_N)
        if use0:
            cute.mbarrier_try_wait_parity(bar0, parity)
        else:
            cute.mbarrier_try_wait_parity(bar1, parity)
        cute.sync_threads()

        for kk in range(TILE_K // 16):
            # A fragment: warp_m's m16 rows
            fa = cute.ldmatrix(
                sAs, (warp_m * 16 + lane16) * TILE_K + kk * 16 + half * 8,
                num=4, trans=False)
            a_frags = [cute.extract_frag(fa, i) for i in range(4)]
            # two n8 atoms across warp_n's 16-column window
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

    # epilogue: full 64-wide tile, block-diagonal in the grid
    CN = TILE_N * 2        # global C row stride (2 CTAs, N direction)
    for na in range(NATOMS_N):
        sub = [acc[na * 4 + i] for i in range(4)]
        col0 = bidx * TILE_N + (warp_n * 4 + na) * 8
        row0 = (bidx * TILE_M + warp_m * 16 + group) * CN + col0 + t2
        mC[row0] = sub[0]
        mC[row0 + 1] = sub[1]
        mC[row0 + 8 * CN] = sub[2]
        mC[row0 + 8 * CN + 1] = sub[3]


@cute.jit
def run_wide_n_gemm(a, b, c, k_tiles: cutlass.Int32):
    wide_n_gemm_kernel(a, b, c, k_tiles).launch(
        grid=(2, 1, 1), block=(32 * NW, 1, 1))


@pytest.mark.sm120
@pytest.mark.parametrize("K", [64, 128, 256])
def test_wide_n_pipelined_gemm(K):
    cutlass.cuda.initialize_cuda_context()
    M = TILE_M * 2
    N = TILE_N * 2
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    ra = TensorMapRecipe(dtype=torch.float16, shape=(M, K),
                         strides_elems=(K, 1), box=(TILE_K, TILE_M))
    rb = TensorMapRecipe(dtype=torch.float16, shape=(K, N),
                         strides_elems=(N, 1), box=(TILE_N, TILE_K))

    run_wide_n_gemm(CUtensorMapView(ra, a), CUtensorMapView(rb, b), c,
                    K // TILE_K)
    torch.cuda.synchronize()

    full = torch.matmul(a.float(), b.float())
    expected = torch.zeros_like(full)
    for bx in range(2):
        expected[bx * TILE_M:(bx + 1) * TILE_M,
                 bx * TILE_N:(bx + 1) * TILE_N] = \
            full[bx * TILE_M:(bx + 1) * TILE_M,
                 bx * TILE_N:(bx + 1) * TILE_N]
    torch.testing.assert_close(c, expected, atol=0.5, rtol=0.5)
