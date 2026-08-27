"""M9 (option B): self-built sm120 warp-mma MLA decode math core.

Absorbed-form FlashMLA decode on the self stack: 16 heads per CTA, streaming
16-token KV tiles with FlashDecoding-style split-KV (sequence dimension
partitioned across CTAs, partial (O, m, l) combined by a second kernel);
QK^T via m16n8k16 f16 mma, online softmax (f32 exp via ex2.approx), PV via
m16n8k16 with the O accumulator held in registers. Golden vs torch; perf vs
the same torch reference (there is NO official CuTeDSL MLA baseline on
sm_120a — hardware boundary, see README).
"""
import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/cutlass_compat"))

import cutlass
import cutlass.cute as cute

D, DV, HTILE = 576, 512, 16
TILE_S = 16
NTHREADS = 128  # 4 warps
DP = D + 8      # padded smem row pitch: 16B-aligned, bank-rotated rows


@cute.kernel
def mla_decode_kernel(mQ: cute.Tensor, mK: cute.Tensor, mP: cute.Tensor,
                      mM: cute.Tensor, mL: cute.Tensor,
                      S: cutlass.Int32, H: cutlass.Constexpr,
                      B: cutlass.Constexpr, scale: cutlass.Constexpr,
                      tps: cutlass.Constexpr, nsplit: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    sQ = cute.make_smem_array("mla_sQ", HTILE * D, element=cutlass.Float16)
    sKt = cute.make_smem_array("mla_sKt", D * TILE_S, element=cutlass.Float16)
    sKn = cute.make_smem_array("mlaKn", TILE_S * DP, element=cutlass.Float16)
    pS = cute.make_smem_array("mla_pS", HTILE * TILE_S,
                              element=cutlass.Float16)
    sS = cute.make_smem_array("mla_sS", HTILE * TILE_S,
                              element=cutlass.Float32)
    sM = cute.make_smem_array("mla_sM", HTILE * 3, element=cutlass.Float32)

    groups = H // HTILE
    bhg = bidx // nsplit          # (batch, head-group) linear index
    sp = bidx % nsplit            # KV split owned by this CTA
    b = bhg // groups
    hg = bhg % groups
    q_base = (b * H + hg * HTILE) * D
    Sn = cute.i32_to_index(S)
    k_base = b * (Sn * D)

    for i in range(HTILE * D // NTHREADS):
        gi = tidx + (i * NTHREADS)
        sQ[gi] = mQ[q_base + gi]

    if tidx < HTILE:
        sM[tidx * 3] = cute.const_f32(-1.0e30)      # running max
        sM[tidx * 3 + 1] = cute.zero_f32()           # running sum
        sM[tidx * 3 + 2] = cute.const_f32(1.0)       # last rescale factor
    cute.sync_threads()

    # O accumulator lives in registers: 16 vcols x 4 f32 per warp
    o_frags = [cute.zero_f32() for _ in range(64)]

    lane = tidx % 32
    wid = tidx // 32
    lane16 = lane % 16
    half8 = (lane // 16) * 8
    row_a = lane // 4
    t2 = (lane % 4) * 2

    sp32 = cute.idx_to_i32(sp)
    lo32 = sp32 * tps
    hi32 = cute.min_i32(lo32 + tps, cute.div_i32(S, TILE_S))
    for t in range(cute.i32_to_index(lo32), cute.i32_to_index(hi32)):
        # ---- load K tile into both mirrors (natural + dim-major).
        # Division-free addressing: D = 576 = 4*128 + 64 — four full
        # NTHREADS-wide passes (d = rep*128 + tidx, tok folded at trace
        # time) plus a power-of-2 tail (64 cols -> div/mod become shifts).
        tile_base = k_base + t * (TILE_S * D)
        for tok in range(TILE_S):
            row = tile_base + tok * D
            for rep in range(D // NTHREADS):
                d = rep * NTHREADS + tidx
                v = mK[row + d]
                sKn[tok * DP + d] = v
                sKt[d * TILE_S + tok] = v
        TAIL = D - (D // NTHREADS) * NTHREADS
        for r in range(TILE_S * TAIL // NTHREADS):
            li = tidx + r * NTHREADS
            tok2 = li // TAIL
            d2 = (D // NTHREADS) * NTHREADS + li % TAIL
            v = mK[tile_base + tok2 * D + d2]
            sKn[tok2 * DP + d2] = v
            sKt[d2 * TILE_S + tok2] = v
        cute.sync_threads()

        # ---- QK^T: warps 0/1 each score 8 token cols, M=16 heads
        if wid < 2:
            ncol0 = wid * 8
            acc = [cute.zero_f32() for _ in range(4)]
            for kb in range(D // 16):
                fa = cute.ldmatrix(sQ, lane16 * D + kb * 16 + half8,
                                   num=4, trans=False)
                a_frags = [cute.extract_frag(fa, j) for j in range(4)]
                fb = cute.ldmatrix(
                    sKt, (kb * 16 + lane16) * TILE_S + ncol0, num=2,
                    trans=True)
                b_frags = [cute.extract_frag(fb, j) for j in range(2)]
                acc = cute.gemm(None, acc, a_frags, b_frags)
            for j in range(4):
                r = row_a + (8 if j >= 2 else 0)
                c = ncol0 + t2 + (1 if j % 2 == 1 else 0)
                sS[r * TILE_S + c] = acc[j] * scale
        cute.sync_threads()

        # ---- softmax bookkeeping (one thread per head)
        if tidx < HTILE:
            mx = sM[tidx * 3]
            for c in range(TILE_S):
                mx = cute.fmax_f32(mx, sS[tidx * TILE_S + c])
            f_old = cute.exp_f32(sM[tidx * 3] - mx)
            sM[tidx * 3] = mx
            sM[tidx * 3 + 1] = sM[tidx * 3 + 1] * f_old
            sM[tidx * 3 + 2] = f_old
        cute.sync_threads()

        # ---- p = exp(s - m): write f16 to pS, keep f32 in sS for the sum
        if wid < 2:
            ncol0 = wid * 8
            for j in range(4):
                r = row_a + (8 if j >= 2 else 0)
                c = ncol0 + t2 + (1 if j % 2 == 1 else 0)
                p_v = cute.exp_f32(sS[r * TILE_S + c] - sM[r * 3])
                sS[r * TILE_S + c] = p_v
                pS[r * TILE_S + c] = cute.fptrunc_f16(p_v)
        cute.sync_threads()
        if tidx < HTILE:
            lsum = sM[tidx * 3 + 1]
            for c in range(TILE_S):
                lsum = lsum + sS[tidx * TILE_S + c]
            sM[tidx * 3 + 1] = lsum
        cute.sync_threads()

        # ---- PV in registers: O = O*f_old + P·V; every warp owns its own
        # 16 vcols (uniform code, no divergence); pS fragment loaded once
        fa = cute.ldmatrix(pS, lane16 * TILE_S + half8, num=4, trans=False)
        a_frags = [cute.extract_frag(fa, j) for j in range(4)]
        fA = sM[row_a * 3 + 2]
        fB = sM[(row_a + 8) * 3 + 2]
        newf = []
        for lvc in range(16):
            vbase = wid * 128 + lvc * 8
            fb = cute.ldmatrix(sKn, lane16 * DP + vbase, num=2, trans=True)
            b_frags = [cute.extract_frag(fb, j) for j in range(2)]
            acc2 = [cute.zero_f32() for _ in range(4)]
            acc2 = cute.gemm(None, acc2, a_frags, b_frags)
            for j in range(4):
                f = fA if j < 2 else fB
                newf.append(o_frags[lvc * 4 + j] * f + acc2[j])
        o_frags = newf
        cute.sync_threads()

    # ---- store RAW partials: O (unnormalized), m, l
    pbase = bidx * (HTILE * DV)
    for lvc in range(16):
        vbase = wid * 128 + lvc * 8
        for j in range(4):
            r = row_a + (8 if j >= 2 else 0)
            c = vbase + t2 + (1 if j % 2 == 1 else 0)
            mP[pbase + r * DV + c] = o_frags[lvc * 4 + j]
    if tidx < HTILE:
        mM[bidx * HTILE + tidx] = sM[tidx * 3]
        mL[bidx * HTILE + tidx] = sM[tidx * 3 + 1]


@cute.kernel
def mla_combine_kernel(mP: cute.Tensor, mM: cute.Tensor, mL: cute.Tensor,
                       mO: cute.Tensor, nsplit: cutlass.Constexpr,
                       H: cutlass.Constexpr):
    """FlashDecoding combine: one CTA per (batch, head) row."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    groups = H // HTILE
    b = bidx // H
    hh = bidx % H
    hg = hh // HTILE
    row = hh % HTILE
    base = (b * groups + hg) * nsplit

    mstar = cute.const_f32(-1.0e30)
    for s in range(nsplit):
        mstar = cute.fmax_f32(mstar, mM[(base + s) * HTILE + row])
    lsum = cute.zero_f32()
    for s in range(nsplit):
        lsum = lsum + mL[(base + s) * HTILE + row] * cute.exp_f32(
            mM[(base + s) * HTILE + row] - mstar)
    for i in range(DV // NTHREADS):
        d = tidx + (i * NTHREADS)
        acc = cute.zero_f32()
        for s in range(nsplit):
            w = cute.exp_f32(mM[(base + s) * HTILE + row] - mstar)
            acc = acc + w * mP[((base + s) * HTILE + row) * DV + d]
        mO[bidx * DV + d] = acc / lsum


@cute.jit
def run_mla_decode(q, k, p, m, l, S: cutlass.Int32, H: cutlass.Constexpr,
                   B: cutlass.Constexpr, scale: cutlass.Constexpr,
                   tps: cutlass.Constexpr, nsplit: cutlass.Constexpr):
    mla_decode_kernel(q, k, p, m, l, S, H, B, scale, tps, nsplit).launch(
        grid=[B * (H // HTILE) * nsplit, 1, 1], block=[NTHREADS, 1, 1])


@cute.jit
def run_mla_combine(p, m, l, o, nsplit: cutlass.Constexpr,
                    H: cutlass.Constexpr, B: cutlass.Constexpr):
    mla_combine_kernel(p, m, l, o, nsplit, H).launch(
        grid=[B * H, 1, 1], block=[NTHREADS, 1, 1])


def torch_mla_ref(q, k, scale):
    """q (B,H,576) f16, k (B,S,576) f16 -> (B,H,512) f32."""
    scores = torch.einsum("bhd,bsd->bhs", q.float(), k.float()) * scale
    p = torch.softmax(scores, dim=-1)
    o = torch.einsum("bhs,bsd->bhd", p, k.float()[:, :, :DV])
    return o


def split_plan(S):
    total = S // TILE_S
    nsplit = max(1, min(32, total // 4))
    tps = (total + nsplit - 1) // nsplit
    return tps, nsplit


def run_case(B, S, H, check=True, iters=20):
    cutlass.cuda.initialize_cuda_context()
    torch.manual_seed(0)
    scale = 1.0 / (D ** 0.5)
    tps, nsplit = split_plan(S)
    groups = H // HTILE
    q = torch.randn(B, H, D, dtype=torch.float16, device="cuda")
    k = torch.randn(B, S, D, dtype=torch.float16, device="cuda")
    o = torch.zeros(B * H * DV, dtype=torch.float32, device="cuda")
    nps = B * groups * nsplit * HTILE
    pP = torch.zeros(nps * DV, dtype=torch.float32, device="cuda")
    pM = torch.zeros(nps, dtype=torch.float32, device="cuda")
    pL = torch.zeros(nps, dtype=torch.float32, device="cuda")

    from cutlass.cute.runtime import from_dlpack
    q_t, k_t, o_t = from_dlpack(q), from_dlpack(k), from_dlpack(o)
    p_t, m_t, l_t = from_dlpack(pP), from_dlpack(pM), from_dlpack(pL)
    c1 = cute.compile(run_mla_decode, q_t, k_t, p_t, m_t, l_t,
                      S, H, B, scale, tps, nsplit)
    c2 = cute.compile(run_mla_combine, p_t, m_t, l_t, o_t,
                      nsplit, H, B)

    def launch():
        c1(q_t, k_t, p_t, m_t, l_t, S, H, B, scale, tps, nsplit)
        c2(p_t, m_t, l_t, o_t, nsplit, H, B)

    launch()
    torch.cuda.synchronize()

    out = o.view(B, H, DV)
    if check:
        ref = torch_mla_ref(q, k, scale)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
        return None
    # perf
    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        launch()
    torch.cuda.synchronize()
    self_us = (time.perf_counter() - t0) / iters * 1e6
    for _ in range(3):
        torch_mla_ref(q, k, scale)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch_mla_ref(q, k, scale)
    torch.cuda.synchronize()
    ref_us = (time.perf_counter() - t0) / iters * 1e6
    return self_us, ref_us


@pytest.mark.sm120
@pytest.mark.parametrize("B,S,H", [
    (1, 1024, 64),
    (2, 2048, 64),
    (1, 4096, 128),
    (4, 512, 128),
])
def test_mla_decode_golden(B, S, H):
    run_case(B, S, H, check=True)


@pytest.mark.sm120
def test_mla_decode_perf():
    rows = []
    for B, S, H in [(1, 1024, 64), (1, 4096, 128), (2, 2048, 128)]:
        r = run_case(B, S, H, check=False)
        rows.append((B, S, H) + r)
        print(f"MLA decode B={B} S={S} H={H}: self {r[0]:.1f}us vs torch-ref "
              f"{r[1]:.1f}us ({r[1]/r[0]:.2f}x)")
    with open("/tmp/mla_perf.txt", "w") as f:
        for row in rows:
            f.write(f"{row}\n")
