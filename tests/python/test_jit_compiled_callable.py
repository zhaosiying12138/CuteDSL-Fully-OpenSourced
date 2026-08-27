"""Argument-rebinding regressions for callables returned by ``cute.compile``."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/cutlass_compat"))

import cutlass
from self_cutedsl.frontend.jit import _CompiledCallable


class _Tensor:
    shape = (1,)

    def __init__(self, address):
        self.address = address

    def data_ptr(self):
        return self.address


class _DType:
    def __init__(self, name):
        self.name = name


class _Scalar:
    def __init__(self, value, dtype):
        self.value = value
        self.dtype = _DType(dtype)


class CUstream:
    def __init__(self, handle):
        self.handle = handle


def test_runtime_scalars_bind_before_trailing_captured_options():
    prime_tensors = [_Tensor(index) for index in range(5)]
    prime = [
        *prime_tensors,
        _Scalar(1, "Int32"),
        _Scalar(1e-6, "Float32"),
        False,
        CUstream(1),
    ]
    compiled = _CompiledCallable(lambda *args: args, prime)
    runtime_tensors = [_Tensor(index + 10) for index in range(5)]

    rebound = compiled(*runtime_tensors, 7, 2e-5)

    assert list(rebound[:5]) == runtime_tensors
    assert rebound[5:7] == (7, 2e-5)
    assert rebound[7] is False
    assert rebound[8] is prime[8]


def test_omitted_constexpr_is_skipped_before_runtime_stream():
    prime = [_Tensor(1), _Tensor(2), 64, CUstream(3)]
    compiled = _CompiledCallable(lambda *args: args, prime)
    new_tensors = (_Tensor(10), _Tensor(20))
    new_stream = CUstream(30)

    rebound = compiled(*new_tensors, new_stream)

    assert rebound[:2] == new_tensors
    assert rebound[2] == 64
    assert rebound[3] is new_stream


def test_typed_literals_fold_and_restore_region_state():
    left = cutlass.Int32(7)
    right = cutlass.Int32(3)

    assert (left + right).value == 10
    assert (left // right).value == 2
    assert bool(left > right)

    snapshot = left.__snapshot__()
    left.ssa = object()
    left.__restore__(snapshot)
    assert left.ssa is None
    assert left.value == 7
