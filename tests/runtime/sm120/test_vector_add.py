"""M1 DoD test: hand-written MLIR vector-add → open PTX (sm_120a) → Driver JIT
→ executed on the RTX 5090, numerically verified.

Runs entirely in the SELF environment: no official CuTeDSL wheel, no nvcc,
no NVRTC, no ptxas. The PTX was produced by cutlass-compiler + LLVM NVPTX.
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

MLIR_SRC = ROOT / "tests/ptx/vector_add.mlir"


def _build_ptx() -> str:
    ptx = compile_mlir_to_ptx(MLIR_SRC)
    assert ".target sm_120a" in ptx
    assert ".version 8.7" in ptx
    assert "add" in entry_names(ptx)
    return ptx


@pytest.mark.sm120
def test_vector_add_end_to_end():
    if not torch.cuda.is_available():
        pytest.fail("SM120 release-gate: GPU required, none present")
    assert "NVIDIA GeForce RTX 5090" in torch.cuda.get_device_name(0)

    ptx = _build_ptx()
    jit = DriverJit(ptx)
    manifest = LaunchManifest(
        entry="add",
        args=[{"name": "a", "type": "ptr"},
              {"name": "b", "type": "ptr"},
              {"name": "c", "type": "ptr"},
              {"name": "n", "type": "i32"}],
        block=(256, 1, 1),
        grid=(16, 1, 1),
    )

    n = 4096
    a = torch.arange(n, device="cuda", dtype=torch.float32)
    b = torch.full((n,), 100.0, device="cuda", dtype=torch.float32)
    c = torch.zeros(n, device="cuda", dtype=torch.float32)

    jit.launch(manifest, a, b, c, n)
    jit.synchronize()

    torch.cuda.synchronize()
    expected = a + b
    assert torch.allclose(c, expected), f"mismatch max={float((c-expected).abs().max())}"
    print("M1_VECTOR_ADD_OK on", torch.cuda.get_device_name(0))
