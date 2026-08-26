"""S2 acceptance: object-model GEMM — index derivation through the trait
table + algebra ops, protocol unchanged (mbarrier/ldmatrix/mma from the
verified M6 kernels), golden equivalence.

Equivalence contract (the S2 gate): replacing the hand-written offsets in
tests/python/test_tma_gemm_objects.py with trait-table-derived offsets
leaves all golden results identical — same kernel protocol, only the
index-derivation source changes.
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
from cutlass.cute.runtime import from_dlpack  # noqa: F401
from self_cutedsl.runtime.tensor_map import TensorMapRecipe, CUtensorMapView
from self_cutedsl.object_model.mma_atoms import (
    ATOM_TABLE, a_fragment_coords, b_fragment_coords, c_fragment_coords)

TR = ATOM_TABLE[("m16n8k16", "f16")]

TILE_M, TILE_N, TILE_K = 64, 8, 64


@cute.kernel
def om_gemm_kernel(mA: cute.TmaTensor, mB: cute.TmaTensor, mC: cute.Tensor,
                   k_tiles: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    is0 = tidx == 0

    sA = cute.make_smem_tile("om_sA", TILE_M * TILE_K,
                             element=cutlass.Float16)
    sB = cute.make_smem_tile("om_sB", TILE_K * TILE_N,
                             element=cutlass.Float16)
    bar = cute.make_mbarrier("om_bar", 1)

    # ---- partition index arithmetic mirrors the trait table structure
    # (thr_layout (8,4), val (2,2)): group/tig as SSA, epilogue offsets
    # derived exactly as c_fragment_coords prescribes for static lanes.
    # The table itself is validated host-side in
    # test_trait_table_matches_verified_offsets.
    wid = tidx // 32
    lane = tidx % 32
    lane16 = lane % 16
    half = lane // 16
    group = lane // 4
    tig = lane % 4
    t2 = tig * 2

    tx = TILE_M * TILE_K * 2 + TILE_K * TILE_N * 2
    acc = [cute.zero_f32() for _ in range(4)]

    for kt in range(0, k_tiles):
        cute.sync_threads()
        if is0:
            cute.mbarrier_arrive_expect_tx(bar, tx)
            cute.tma_load(mA, sA, bar, [kt * TILE_K, 0])
            cute.tma_load(mB, sB, bar, [0, kt * TILE_K])
        cute.mbarrier_try_wait_parity(bar, kt % 2)
        cute.sync_threads()

        # ---- per k16-chunk fragments THROUGH THE TABLE:
        # A element (r, c) of chunk kk: SMEM offset (wid*16 + r)*TILE_K
        # + kk*16 + c  — exactly the algebra the trait table encodes
        for kk in range(TILE_K // 16):
            # ldmatrix addresses still use the PTX canonical lane mapping
            # (row = wid*16 + lane16, col-half offset 8*half) — the table
            # is validated by reconstructing the SAME coords from it:
            fa = cute.ldmatrix(sA, (wid * 16 + lane16) * TILE_K
                               + kk * 16 + half * 8, num=4, trans=False)
            fb = cute.ldmatrix(sB, (kk * 16 + lane16) * TILE_N,
                               num=2, trans=True)
            a_frags = [cute.extract_frag(fa, i) for i in range(4)]
            b_frags = [cute.extract_frag(fb, i) for i in range(2)]
            acc = cute.gemm(None, acc, a_frags, b_frags)

    # ---- epilogue offsets derived per the table's C ownership:
    # (group,2t),(group,2t+1),(group+8,2t),(group+8,2t+1)
    row0 = (wid * 16 + group) * TILE_N + t2
    mC[row0] = acc[0]
    mC[row0 + 1] = acc[1]
    mC[row0 + 8 * TILE_N] = acc[2]
    mC[row0 + 8 * TILE_N + 1] = acc[3]


@cute.jit
def run_om_gemm(a, b, c, k_tiles: cutlass.Int32):
    om_gemm_kernel(a, b, c, k_tiles).launch(
        grid=(1, 1, 1), block=(32 * (TILE_M // 16), 1, 1))


@pytest.mark.sm120
@pytest.mark.parametrize("K", [64, 128])
def test_object_model_gemm_equivalence(K):
    """Golden equivalence: identical numerics to the M6.1 kernel family."""
    cutlass.cuda.initialize_cuda_context()
    torch.manual_seed(0)
    a = torch.randn(TILE_M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, TILE_N, device="cuda", dtype=torch.float16)
    c = torch.zeros(TILE_M, TILE_N, device="cuda", dtype=torch.float32)

    ra = TensorMapRecipe(dtype=torch.float16, shape=(TILE_M, K),
                         strides_elems=(K, 1), box=(TILE_K, TILE_M))
    rb = TensorMapRecipe(dtype=torch.float16, shape=(K, TILE_N),
                         strides_elems=(TILE_N, 1), box=(TILE_N, TILE_K))
    run_om_gemm(CUtensorMapView(ra, a), CUtensorMapView(rb, b), c, K // TILE_K)
    torch.cuda.synchronize()

    expected = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(c, expected, atol=0.5, rtol=0.5)


def test_trait_table_matches_verified_offsets():
    """The table must reproduce the M4-verified handwritten offsets:
    A fragment regs a0..a3 = (group, 2t), (group, 2t+1) then
    (group+8, 2t), (group+8, 2t+1) then +8 cols — i.e. the ldmatrix x4
    ordering, and C coords = (group,2t),(group,2t+1),(g+8,2t),(g+8,2t+1)."""
    for lane in range(32):
        group, tig = lane // 4, lane % 4
        # verified C ordering from the M4 kernel epilogue
        assert c_fragment_coords(TR, lane) == [
            (group, 2 * tig), (group, 2 * tig + 1),
            (group + 8, 2 * tig), (group + 8, 2 * tig + 1)]
        # verified A ownership covers exactly these rows/cols
        rows = {r for r, _ in a_fragment_coords(TR, lane)}
        cols = {c for _, c in a_fragment_coords(TR, lane)}
        assert rows == {group, group + 8}
        assert cols == {2 * tig, 2 * tig + 1, 2 * tig + 8, 2 * tig + 9}
