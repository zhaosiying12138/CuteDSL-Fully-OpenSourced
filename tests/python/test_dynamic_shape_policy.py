"""P5 shape-polymorphic policy: marked extents leave the specialization key.

mark_layout_dynamic() used to be a no-op, forcing one re-trace + recompile
per shape. With the policy layer, marked extents are bucketed out of the
jit cache key and published as runtime scalars — one compiled plan is
shared across their values, with correctness carried by the runtime
bound (the same channel the MLA decode kernel uses for its sequence
length).

Contract (documented on TensorMeta.mark_layout_dynamic): marked extents
must not be baked into addressing arithmetic. The grid-stride kernel here
satisfies it — the extent only feeds the dynamic loop bound.
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
from cutlass.torch import from_dlpack

from self_cutedsl.frontend import jit as J


@cute.kernel
def flat_add(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor,
             n: cutlass.Int32, scale: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bdim_x, _, _ = cute.arch.block_dim()
    bidx, _, _ = cute.arch.block_idx()
    flat = bidx * bdim_x + tidx
    stride = bdim_x * 1
    for i in range(flat, n, stride):
        c[i] = a[i] + b[i] * scale


@cute.jit
def launch_flat_add(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor,
                    n: cutlass.Int32, scale: cutlass.Constexpr):
    flat_add(a, b, c, n, scale).launch(grid=(64, 1, 1), block=(128, 1, 1))


class _CompileCounter:
    def __init__(self):
        self.orig = J.compile_mlir_to_ptx
        self.count = 0

    def __enter__(self):
        def counted(*a, **k):
            self.count += 1
            return self.orig(*a, **k)
        J.compile_mlir_to_ptx = counted
        return self

    def __exit__(self, *exc):
        J.compile_mlir_to_ptx = self.orig


@pytest.mark.sm120
@pytest.mark.parametrize("mark", [False, True])
def test_shape_polymorphic_policy(mark):
    cutlass.cuda.initialize_cuda_context()
    shapes = [1000, 4096, 123457]

    def make(n):
        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)
        return a, b, torch.empty_like(a)

    with _CompileCounter() as cc:
        for n in shapes:
            a, b, c = make(n)
            av = from_dlpack(a)
            bv = from_dlpack(b)
            cv = from_dlpack(c)
            if mark:
                av = av.mark_layout_dynamic()
                bv = bv.mark_layout_dynamic()
                cv = cv.mark_layout_dynamic()
            launch_flat_add(av, bv, cv, cutlass.Int32(n), 2.0)
            torch.cuda.synchronize()
            assert torch.equal(c, a + 2.0 * b), n

    if mark:
        assert cc.count == 1, (
            f"marked tensors should share ONE compiled plan, got {cc.count}")
    else:
        assert cc.count == len(shapes), (
            f"unmarked tensors specialize per shape (expected "
            f"{len(shapes)} compiles), got {cc.count}")
