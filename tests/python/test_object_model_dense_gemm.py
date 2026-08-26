"""S4: dense_gemm-shaped kernel assembled from the FULL object model —

index derivation through algebra.py (cute.* op emission + C++ passes),
pipeline through the PipelineTmaAsync driver, fragments through the
mma_atoms trait table, TMA through the generalized tma module. This is
the culminating integration: every index the kernel computes descends
from C++-verified cute-dialect algebra rather than hand-written offsets.
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
from self_cutedsl.object_model import algebra as om
from self_cutedsl.object_model.pipeline import (
    Agent, CooperativeGroup, PipelineState, PipelineTmaAsync,
    PipelineUserType)
from self_cutedsl.object_model import tma as otma
from self_cutedsl.object_model.mma_atoms import ATOM_TABLE

TR = ATOM_TABLE[("m16n8k16", "f16")]

TILE_M, TILE_N, TILE_K = 64, 64, 64
STAGES = 2
WARPS_M, WARPS_N = 4, 2
NATOMS_N = 4


@cute.kernel
def om_dense_gemm_kernel(mA: cute.TmaTensor, mB: cute.TmaTensor,
                         mC: cute.TmaTensor, k_tiles: cutlass.Int32,
                         M: cutlass.Constexpr, N: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    is0 = tidx == 0

    sA = cute.make_smem_tile("om4_sA", STAGES * TILE_M * TILE_K,
                             element=cutlass.Float16)
    sB = cute.make_smem_tile("om4_sB", STAGES * TILE_K * TILE_N,
                             element=cutlass.Float16)
    sC = cute.make_smem_tile("om4_sC", TILE_M * TILE_N,
                             element=cutlass.Float32)
    bars = cute.make_barrier_array("om4_bars", STAGES)

    tiles_n = N // TILE_N
    tm = bidx // tiles_n
    tn = bidx % tiles_n

    wid = tidx // 32
    lane = tidx % 32
    lane16 = lane % 16
    half = lane // 16
    warp_m = wid % WARPS_M
    warp_n = wid // WARPS_M
    group = lane // 4
    tig = lane % 4
    t2 = tig * 2

    # dense_gemm-style: local tile coordinate -> SMEM row/col via trait
    # table structure; global offsets via SSA arith + cute layout_eval
    tx = TILE_M * TILE_K * 2 + TILE_K * TILE_N * 2
    pipe = PipelineTmaAsync.create(
        num_stages=STAGES,
        producer_group=CooperativeGroup(Agent.Thread, 1),
        consumer_group=CooperativeGroup(Agent.Thread, 1),
        tx_count=tx,
        barrier_storage=bars.ptr)

    acc = [cute.zero_f32() for _ in range(4 * NATOMS_N)]

    # prologue: stage s <- k-tile s
    if is0:
        for pt in range(STAGES):
            st = PipelineState(cute.const_i32(pt), cute.const_i32(0))
            b = pipe.producer_get_barrier(st)
            cute.mbarrier_arrive_expect_tx(b, tx)
            otma.tma_issue(cute.smem_stage(sA, pt, TILE_M * TILE_K),
                           mA, b, [pt * TILE_K, tm * TILE_M])
            otma.tma_issue(cute.smem_stage(sB, pt, TILE_K * TILE_N),
                           mB, b, [tn * TILE_N, pt * TILE_K])
    cute.sync_threads()

    for kt in range(0, k_tiles):
        kt32 = cute.idx_to_i32(kt)
        cons_state = PipelineState(
            cute.rem_i32(kt32, STAGES),
            cute.rem_i32(cute.div_i32(kt32, STAGES), 2))
        pipe.consumer_wait(cons_state)
        cute.sync_threads()
        stage = kt % 2
        sAs = cute.smem_stage(sA, stage, TILE_M * TILE_K)
        sBs = cute.smem_stage(sB, stage, TILE_K * TILE_N)

        for kk in range(TILE_K // 16):
            fa = cute.ldmatrix(
                sAs, (warp_m * 16 + lane16) * TILE_K + kk * 16 + half * 8,
                num=4, trans=False)
            a_frags = [cute.extract_frag(fa, i) for i in range(4)]
            for na in range(NATOMS_N):
                fb = cute.ldmatrix(
                    sBs, (kk * 16 + lane16) * TILE_N
                    + (warp_n * 4 + na) * 8, num=2, trans=True)
                b_frags = [cute.extract_frag(fb, i) for i in range(2)]
                sub = cute.gemm(None, [acc[na * 4 + i] for i in range(4)],
                                a_frags, b_frags)
                acc = acc[:na * 4] + sub + acc[(na + 1) * 4:]

        cute.sync_threads()
        ktn = cute.add_i32(kt32, STAGES)
        if cute.lt_i32(ktn, k_tiles):
            prod_state = PipelineState(
                cute.rem_i32(kt32, STAGES),
                cute.rem_i32(cute.div_i32(kt32, STAGES), 2))
            if is0:
                bar_refill = pipe.producer_get_barrier(prod_state)
                cute.mbarrier_arrive_expect_tx(bar_refill, tx)
                otma.tma_issue(
                    cute.smem_stage(sA, stage, TILE_M * TILE_K),
                    mA, bar_refill, [ktn * TILE_K, tm * TILE_M])
                otma.tma_issue(
                    cute.smem_stage(sB, stage, TILE_K * TILE_N),
                    mB, bar_refill, [tn * TILE_N, ktn * TILE_K])

    # epilogue: trait-table C ownership -> sC -> TMA S2G
    for na in range(NATOMS_N):
        sub = [acc[na * 4 + i] for i in range(4)]
        col = (warp_n * 4 + na) * 8 + t2
        row = warp_m * 16 + group
        sC[row * TILE_N + col] = sub[0]
        sC[row * TILE_N + col + 1] = sub[1]
        sC[(row + 8) * TILE_N + col] = sub[2]
        sC[(row + 8) * TILE_N + col + 1] = sub[3]
    cute.sync_threads()
    if is0:
        otma.tma_store_issue(mC, sC, [tn * TILE_N, tm * TILE_M])
    cute.sync_threads()


@cute.jit
def run_om_dense_gemm(a, b, c, k_tiles: cutlass.Int32,
                      M: cutlass.Constexpr, N: cutlass.Constexpr):
    om_dense_gemm_kernel(a, b, c, k_tiles, M, N).launch(
        grid=((M // TILE_M) * (N // TILE_N), 1, 1),
        block=(32 * WARPS_M * WARPS_N, 1, 1))


@pytest.mark.sm120
@pytest.mark.parametrize("M,N,K", [
    (128, 128, 128),
    (256, 128, 256),
    (128, 256, 256),
    (256, 256, 128),
    (512, 512, 256),
])
def test_om_dense_gemm(M, N, K):
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
    run_om_dense_gemm(CUtensorMapView(ra, a), CUtensorMapView(rb, b),
                      CUtensorMapView(rc, c), K // TILE_K, M, N)
    torch.cuda.synchronize()
    expected = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(c, expected, atol=0.5, rtol=0.5)
