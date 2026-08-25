"""M4: warp mma.sync m16n8k16 f16->f32 GEMM on the 5090, numerically verified.

The kernel is hand-written MLIR (tests/ptx/mma_m16n8k16.mlir) loading
operands from gmem in the PTX-documented register layout and issuing a real
mma.sync; golden check against torch.matmul in f32.
"""
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from self_cutedsl.compiler import compile_mlir_to_ptx, entry_names  # noqa: E402
from self_cutedsl.runtime import DriverJit, LaunchManifest  # noqa: E402

MLIR_SRC = ROOT / "tests/ptx/mma_ldmatrix.mlir"


@pytest.mark.sm120
def test_mma_m16n8k16():
    if not torch.cuda.is_available():
        pytest.fail("SM120 release-gate: GPU required")

    ptx = compile_mlir_to_ptx(MLIR_SRC)
    assert "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32" in ptx, \
        "PTX must contain the real warp MMA instruction"
    assert "ldmatrix.sync.aligned.m8n8.x4.shared.b16" in ptx
    assert "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16" in ptx

    jit = DriverJit(ptx)
    manifest = LaunchManifest(
        entry="mma16816",
        args=[{"name": "a", "type": "ptr"},
              {"name": "b", "type": "ptr"},
              {"name": "d", "type": "ptr"}],
        grid=(1, 1, 1),
        block=(32, 1, 1),
    )

    torch.manual_seed(0)
    a = torch.randn(16, 16, device="cuda", dtype=torch.float16)
    b_t = torch.randn(8, 16, device="cuda", dtype=torch.float16)   # [n][k]
    b = b_t.t().contiguous()                                       # [k][n] flat k*8+n
    d = torch.zeros(16, 8, device="cuda", dtype=torch.float32)

    jit.launch(manifest, a, b, d)
    jit.synchronize()

    expected = torch.matmul(a.float(), b_t.t().float())  # A[16,16] @ B[k,n] -> [16,8]
    torch.testing.assert_close(d, expected, atol=1e-2, rtol=1e-2)
    print("M4_MMA_OK:", torch.cuda.get_device_name(0))
