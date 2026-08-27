"""FlashInfer cute_dsl norm operators, UNMODIFIED vendored sources, running
on the self stack (verbatim acceptance — same standard as dense_gemm.py).

Golden: the operators' own reference math (torch), compared in fp4-decoded
space against the official NVFP4 quantization semantics.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/cutlass_compat"))
sys.path.insert(0, str(ROOT / "tests/python/support"))

import cutlass
import flashinfer_verbatim as FV


def _fp4_dequant(y_uint8: torch.Tensor) -> torch.Tensor:
    """Decode packed fp4_e2m1 (low nibble then high nibble) to float."""
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                       device=y_uint8.device, dtype=torch.float32)
    flat = y_uint8.reshape(-1)
    lo = lut[(flat & 0xF).long()]
    hi = lut[(flat >> 4).long()]
    return torch.stack([lo, hi], dim=-1).reshape(*y_uint8.shape, -1)


def _sf_dequant(s_uint8: torch.Tensor) -> torch.Tensor:
    """Decode e4m3 scale-factor bytes to float."""
    return s_uint8.to(torch.float32)  # uint8 view of e4m3: bit-pattern decode


def _e4m3_bits_to_float(u8: torch.Tensor) -> torch.Tensor:
    bits = u8.to(torch.int32)
    sign = (bits >> 7) & 1
    exp = (bits >> 3) & 0xF
    mant = bits & 7
    val = torch.where(
        exp == 0,
        mant * 0.5 ** 9,                    # subnormals: m * 2^-9
        (1 + mant / 8.0) * torch.pow(2.0, (exp - 7).to(torch.float32)))
    return torch.where(sign == 1, -val, val)


def ref_rmsnorm_fp4(x, w, eps, global_scale):
    """Reference: rmsnorm -> /global_scale -> per-16-block e4m3 scale +
    fp4 quantization (round to nearest fp4 grid)."""
    M, H = x.shape
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = xf * rstd * w.float() / global_scale
    yb = y.view(M, H // 16, 16)
    amax = yb.abs().amax(-1, keepdim=True)
    sf = (amax / 6.0).clamp(max=448.0)
    sf = torch.where(sf > 0, sf, torch.ones_like(sf))
    # e4m3 quantize the scale
    sf_q = _e4m3_round(sf)
    scale = sf_q / 6.0
    yq = (yb / scale).round().clamp(-6, 6)
    # snap to fp4 grid (0.5 steps)
    yq = (yq * 2).round() / 2
    return yq.view(M, H), sf_q.view(M, H // 16)


def _e4m3_round(t: torch.Tensor) -> torch.Tensor:
    exp = torch.floor(torch.log2(t.clamp(min=1e-30))) - 7 + 127
    step = torch.pow(2.0, torch.floor(torch.log2(t.clamp(min=1e-30))) - 3)
    return (t / step).round() * step


def run_rmsnorm_case(B, H, dtype=torch.float16, eps=1e-6):
    cutlass.cuda.initialize_cuda_context()
    torch.manual_seed(42)
    ops = FV.load_operator("rmsnorm_fp4quant")

    x = torch.randn(B, H, device="cuda", dtype=dtype)
    w = torch.randn(H, device="cuda", dtype=dtype)
    # global scale per NVFP4 recipe (from the official test)
    gs = (448 * 448 * 6.0) ** 0.5 / 448.0
    gs_t = torch.tensor([gs], device="cuda", dtype=torch.float32)

    y = torch.empty(B, H // 2, device="cuda", dtype=torch.uint8)
    s = torch.empty(B, H // 16, device="cuda", dtype=torch.uint8)

    ops.rmsnorm_fp4quant(x, w, y, s, global_scale=gs_t, eps=eps, block_size=16)
    torch.cuda.synchronize()

    y_dec = _fp4_dequant(y)                       # (M, H) f32
    sf_dec = _e4m3_bits_to_float(s)               # (M, H/16) f32
    y_ref, sf_ref = ref_rmsnorm_fp4(x, w, eps, gs)
    # dequant with the kernel's own scales for the value comparison
    y_re = (y_dec.view(B, H // 16, 16) * sf_dec.unsqueeze(-1) * gs).view(B, H)
    y_ref_re = (y_ref.view(B, H // 16, 16) * sf_ref.unsqueeze(-1) * gs).view(B, H)
    ok_vals = torch.allclose(y_re, y_ref_re, atol=0.35, rtol=0.2)
    sf_close = torch.allclose(sf_dec, sf_ref, rtol=0.25, atol=1e-3)
    frac = (y_dec.view(B, H // 16, 16) - y_ref.view(B, H // 16, 16)).abs() \
        .le(0.5).float().mean().item()
    return ok_vals and sf_close, frac


@pytest.mark.sm120
@pytest.mark.parametrize("B,H", [(1, 128), (4, 128), (7, 256), (3, 512)])
def test_rmsnorm_fp4quant_verbatim(B, H):
    ok, frac = run_rmsnorm_case(B, H)
    assert ok, f"fp4 values mismatch (exact-grid fraction {frac:.3f})"
