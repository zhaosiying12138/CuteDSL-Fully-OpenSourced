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
    return torch.stack([lo, hi], dim=-1).reshape(
        *y_uint8.shape[:-1], y_uint8.shape[-1] * 2
    )


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


def _nearest_e2m1(values: torch.Tensor) -> torch.Tensor:
    """Round to the non-uniform E2M1 grid used by the PTX converter."""
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=values.device,
        dtype=torch.float32,
    )
    return lut[(values.unsqueeze(-1) - lut).abs().argmin(dim=-1)]


def ref_rmsnorm_fp4(x, w, eps, global_scale):
    """True RMSNorm plus the operator's documented half2 NVFP4 recipe."""
    M, H = x.shape
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    rms = (xf * rstd * w.float()).to(x.dtype).float()
    # The source deliberately uses half2 multiplication for x*w, followed by
    # f32 scaling with rstd. Mirror that rounding for exact FP4/SF codes while
    # retaining ``rms`` above as the independent true-math comparison.
    quant_input = (x * w).float() * rstd
    yb = quant_input.view(M, H // 16, 16)
    sf = (yb.abs().amax(-1) * global_scale / 6.0).clamp(max=448.0)
    sf_q = sf.to(torch.float8_e4m3fn).float()
    scaled = torch.where(
        sf_q.unsqueeze(-1) > 0,
        yb * global_scale / sf_q.unsqueeze(-1),
        torch.zeros_like(yb),
    )
    yq = _nearest_e2m1(scaled)
    return rms, yq.view(M, H), sf_q


def run_rmsnorm_case(B, H, dtype=torch.float16, eps=1e-6):
    cutlass.cuda.initialize_cuda_context()
    torch.manual_seed(42)
    ops = FV.load_operator("rmsnorm_fp4quant")

    x = torch.randn(B, H, device="cuda", dtype=dtype)
    w = torch.randn(H, device="cuda", dtype=dtype)
    # Fixed non-trivial NVFP4 global scale; dequantization must reverse it.
    gs = (448 * 448 * 6.0) ** 0.5 / 448.0
    gs_t = torch.tensor([gs], device="cuda", dtype=torch.float32)

    y = torch.empty(B, H // 2, device="cuda", dtype=torch.uint8)
    s = torch.empty(B, H // 16, device="cuda", dtype=torch.uint8)

    ops.rmsnorm_fp4quant(x, w, y, s, global_scale=gs_t, eps=eps, block_size=16)
    torch.cuda.synchronize()

    y_dec = _fp4_dequant(y)                        # (M, H) f32
    sf_dec = _e4m3_bits_to_float(s)                # (M, H/16) f32
    rms_ref, y_ref, sf_ref = ref_rmsnorm_fp4(x, w, eps, gs)

    # Exact semantic checks for both quantized outputs. Negative and positive
    # zero compare equal, which is intentional at the mathematical level.
    frac = y_dec.eq(y_ref).float().mean().item()
    fp4_exact = frac == 1.0
    sf_exact = torch.equal(sf_dec, sf_ref)

    # The upstream operator validates the dequantized result against true
    # RMSNorm math. During quantization the block scale includes global_scale,
    # so dequantization divides by it.
    y_dequant = (
        y_dec.view(B, H // 16, 16) * sf_dec.unsqueeze(-1) / gs
    ).view(B, H)
    diff = (y_dequant - rms_ref).abs()
    rel_diff = diff / (rms_ref.abs() + 1e-8)
    tight_fraction = ((diff <= 0.5) | (rel_diff <= 0.3)).float().mean().item()
    loose_all = bool(((diff <= 2.0) | (rel_diff <= 0.5)).all())
    return fp4_exact and sf_exact and tight_fraction >= 0.99 and loose_all, frac


@pytest.mark.sm120
@pytest.mark.parametrize("B,H", [(1, 128), (4, 128), (7, 256), (3, 512)])
def test_rmsnorm_fp4quant_verbatim(B, H):
    ok, frac = run_rmsnorm_case(B, H)
    assert ok, f"fp4 values mismatch (exact-grid fraction {frac:.3f})"
