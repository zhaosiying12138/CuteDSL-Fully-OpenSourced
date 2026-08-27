"""Verbatim FlashInfer B12x MoE acceptance on the self CuTeDSL stack.

The vendored operator is loaded without source edits.  The goldens below
independently encode/decode NVFP4, including the fast-path scale reciprocal
semantics used by the operator, then evaluate the routed MoE in PyTorch.
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


_E2M1_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _fast_scale_decode(sf8: torch.Tensor) -> torch.Tensor:
    """Mirror the vendored ``fp8_e4m3_to_f32_and_rcp`` scale decoder.

    Its nonzero exponent-zero codes use the same implicit-one expression as
    normal codes when selecting E2M1 values.  MMA dequantization still uses
    the actual E4M3 value, so packing and dequant scales are intentionally
    distinct here.
    """
    bits = sf8.view(torch.uint8).to(torch.int32)
    exponent = (bits >> 3) & 0xF
    mantissa = bits & 7
    decoded = torch.pow(2.0, (exponent - 7).float()) * (1.0 + mantissa / 8.0)
    return torch.where(bits == 0, torch.zeros_like(decoded), decoded)


def _quantize_nvfp4_fast(values: torch.Tensor, global_scale: float = 1.0):
    """Independent fast NVFP4 pack plus true hardware-scale dequantization."""
    shape = values.shape
    blocks = values.float().reshape(*shape[:-1], shape[-1] // 16, 16)
    scale = (blocks.abs().amax(-1) * global_scale / 6.0).clamp(max=448.0)
    scale_e4m3 = scale.to(torch.float8_e4m3fn)
    packing_scale = _fast_scale_decode(scale_e4m3)
    scaled = torch.where(
        packing_scale.unsqueeze(-1) != 0,
        blocks * global_scale / packing_scale.unsqueeze(-1),
        torch.zeros_like(blocks),
    )

    lut = torch.tensor(_E2M1_VALUES, device=values.device)
    codes = (scaled.unsqueeze(-1) - lut).abs().argmin(-1).to(torch.uint8)
    codes = codes.reshape(*shape)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    dequant = (
        lut[codes.long()]
        * scale_e4m3.float().repeat_interleave(16, dim=-1)
        / global_scale
    )
    return packed, scale_e4m3.view(torch.uint8), dequant


def _mma_scale_layout(logical: torch.Tensor, rows: int, k: int, groups: int):
    """Map logical [group,row,k/16] bytes to the official six-mode view."""
    m_tiles = (rows + 127) // 128
    k_blocks = k // 16
    k_tiles = (k_blocks + 3) // 4
    storage = torch.zeros(
        groups,
        m_tiles,
        k_tiles,
        32,
        4,
        4,
        device=logical.device,
        dtype=torch.uint8,
    )
    for row in range(rows):
        for block in range(k_blocks):
            storage[
                :, row // 128, block // 4, row % 32,
                (row % 128) // 32, block % 4,
            ] = logical[:, row, block]
    return storage.view(torch.float8_e4m3fn).permute(3, 4, 1, 5, 2, 0)


def _constant_mma_scales(rows: int, k: int, values: torch.Tensor):
    groups = values.numel()
    bits = values.to(torch.float8_e4m3fn).view(torch.uint8)
    logical = bits[:, None, None].expand(groups, rows, k // 16).clone()
    return _mma_scale_layout(logical, rows, k, groups)


def _run_moe(ops, x, w1, s1, w2, s2, ids, routes):
    experts = w1.shape[0]
    alpha = torch.ones(experts, device=x.device, dtype=torch.float32)
    return ops.b12x_fused_moe(
        x,
        w1,
        s1,
        w2,
        s2,
        ids,
        routes,
        experts,
        ids.shape[1],
        w1_alpha=alpha,
        w2_alpha=alpha,
        fc2_input_scale=torch.ones(1, device=x.device, dtype=torch.float32),
        quant_mode="nvfp4",
    )


@pytest.fixture(scope="module")
def b12x_ops():
    cutlass.cuda.initialize_cuda_context()
    return FV.load_b12x_operator()


def _assert_uniform_routing_exact(b12x_ops, tokens):
    experts, hidden, intermediate = 2, 256, 128
    x = torch.full(
        (tokens, hidden), 0.1, device="cuda", dtype=torch.bfloat16
    )
    expert_codes = torch.tensor([0x11, 0x22], device="cuda", dtype=torch.uint8)
    w1 = expert_codes[:, None, None].expand(
        experts, 2 * intermediate, hidden // 2
    ).clone()
    w2 = expert_codes[:, None, None].expand(
        experts, hidden, intermediate // 2
    ).clone()
    scale_values = torch.full(
        (experts,), 0.015625, device="cuda", dtype=torch.float32
    )
    s1 = _constant_mma_scales(2 * intermediate, hidden, scale_values)
    s2 = _constant_mma_scales(hidden, intermediate, scale_values)
    ids = (torch.arange(tokens, device="cuda", dtype=torch.int32) % experts)[:, None]
    routes = torch.ones(tokens, 1, device="cuda", dtype=torch.float32)

    out = _run_moe(b12x_ops, x, w1, s1, w2, s2, ids, routes)
    torch.cuda.synchronize()

    _, _, x_dequant = _quantize_nvfp4_fast(x)
    lut = torch.tensor(_E2M1_VALUES, device="cuda")
    expected_by_expert = []
    for expert in range(experts):
        weight_value = lut[(expert_codes[expert] & 0xF).long()] * scale_values[expert]
        fc1 = (x_dequant[0] * weight_value).sum()
        activation = (
            torch.nn.functional.silu(fc1) * fc1
        ).to(torch.bfloat16).float()
        activation_row = activation.expand(1, intermediate)
        _, _, activation_dequant = _quantize_nvfp4_fast(activation_row)
        result = (activation_dequant[0] * weight_value).sum().to(torch.bfloat16)
        expected_by_expert.append(result)
    expected_values = torch.stack(expected_by_expert)
    expected = expected_values[ids[:, 0].long(), None].expand_as(out)

    assert torch.equal(out, expected)
    assert not torch.equal(expected_values[0], expected_values[1])


@pytest.mark.sm120
@pytest.mark.parametrize("tokens", [64, 256])
def test_b12x_static_routing_uniform_exact(b12x_ops, tokens):
    """Static routing selects experts for partial and full M tiles exactly."""
    _assert_uniform_routing_exact(b12x_ops, tokens=tokens)


@pytest.mark.sm120
def test_b12x_micro_routing_uniform_exact(b12x_ops):
    """Tiny decode routing uses the 64x128 micro tile exactly."""
    _assert_uniform_routing_exact(b12x_ops, tokens=8)


@pytest.mark.sm120
def test_b12x_dynamic_routing_uniform_exact(b12x_ops):
    """The smallest dynamic-cutover case routes and computes exactly."""
    _assert_uniform_routing_exact(b12x_ops, tokens=641)


@pytest.mark.sm120
def test_b12x_static_routing_random_true_math(b12x_ops):
    """Random encoded weights compare with independent routed MoE math."""
    torch.manual_seed(123)
    tokens, experts, hidden, intermediate = 64, 2, 256, 128
    x = (torch.randn(tokens, hidden, device="cuda") * 0.2).to(torch.bfloat16)
    w1_source = (
        torch.randn(experts, 2 * intermediate, hidden, device="cuda") * 0.15
    ).to(torch.bfloat16)
    w2_source = (
        torch.randn(experts, hidden, intermediate, device="cuda") * 0.15
    ).to(torch.bfloat16)
    w1, w1_scale, w1_dequant = _quantize_nvfp4_fast(w1_source)
    w2, w2_scale, w2_dequant = _quantize_nvfp4_fast(w2_source)
    s1 = _mma_scale_layout(w1_scale, 2 * intermediate, hidden, experts)
    s2 = _mma_scale_layout(w2_scale, hidden, intermediate, experts)
    ids = (torch.arange(tokens, device="cuda", dtype=torch.int32) % experts)[:, None]
    routes = torch.ones(tokens, 1, device="cuda", dtype=torch.float32)

    out = _run_moe(b12x_ops, x, w1, s1, w2, s2, ids, routes)
    torch.cuda.synchronize()

    _, _, x_dequant = _quantize_nvfp4_fast(x)
    reference = torch.empty(tokens, hidden, device="cuda", dtype=torch.float32)
    for token in range(tokens):
        expert = int(ids[token, 0])
        fc1 = x_dequant[token] @ w1_dequant[expert].T
        up, gate = fc1[:intermediate], fc1[intermediate:]
        activation = (
            torch.nn.functional.silu(gate) * up
        ).to(torch.bfloat16).float()
        _, _, activation_dequant = _quantize_nvfp4_fast(activation[None, :])
        reference[token] = activation_dequant[0] @ w2_dequant[expert].T

    actual = out.float()
    difference = (actual - reference).abs()
    relative = difference / (reference.abs() + 1e-6)
    cosine = torch.nn.functional.cosine_similarity(
        actual.reshape(1, -1), reference.reshape(1, -1)
    ).item()
    normalized_rmse = (
        torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference)
    ).item()
    tight_fraction = (
        (difference <= 0.05) | (relative <= 0.25)
    ).float().mean().item()

    assert difference.mean().item() <= 0.02
    assert normalized_rmse <= 0.10
    assert cosine >= 0.995
    assert tight_fraction >= 0.985
