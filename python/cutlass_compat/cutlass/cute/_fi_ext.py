"""flashinfer-operator bridge extensions for the compat cute surface.

Pieces the (unmodified) flashinfer cute_dsl operators need beyond what the
dense_gemm/blockscaled object model already provides. Everything here lowers
through the self stack's textual emitter.
"""
from __future__ import annotations

import enum

from cutlass._bridge_helpers import _emitter, TypedScalar
from cutlass import dtypes as _dt


# ---------------------------------------------------------------------------
# markers / enums
# ---------------------------------------------------------------------------
class TensorSSA:
    """Annotation marker for register-fragment values (official name)."""


class ReductionOp(enum.Enum):
    ADD = 0
    MAX = 1
    MIN = 2
    MUL = 3


class Pointer:
    """Wraps an smem/gmem pointer SSA; toint() gives the raw address."""

    def __init__(self, ssa, space=3):
        self.ssa = ssa
        self.space = space

    def toint(self, loc=None, ip=None):
        e = _emitter()
        t = "i32" if self.space == 3 else "i64"
        op = "llvm.ptrtoint" if self.space == 3 else "llvm.ptrtoint"
        from cutlass._mlir.ir import IrValue
        return IrValue(e.ssa(t, f"{op} {self.ssa.name} : "
                                f"!llvm.ptr<{self.space}> to {t}"))

    def ir_value(self, loc=None, ip=None):
        from cutlass._mlir.ir import IrValue
        return IrValue(self.ssa)


# ---------------------------------------------------------------------------
# small host-side helpers
# ---------------------------------------------------------------------------
def ceil_div(a, b):
    if hasattr(a, "_expr") or a.__class__.__name__ == "DynamicHostValue":
        return a._expr("ceil_div", b)
    if hasattr(a, "fn"):  # DynGridExpr arithmetic chains
        from self_cutedsl.frontend.jit import DynGridExpr

        def _i(v):
            if hasattr(v, "value") and hasattr(v, "ssa"):
                return int(v.value)
            return int(v)

        return DynGridExpr(lambda v, f=a, bb=int(b), _i=_i: -(-_i(f(v)) // bb),
                           f"ceil_div({a!r},{b})")
    return -(-a // b)  # Python ints at trace time


class SymInt(int):
    """Symbolic dynamic size: behaves as the concrete value it was created
    with at trace time (specialization), and is re-bound at call time."""

    def __new__(cls, v=1):
        return super().__new__(cls, v)


def sym_int(v=1):
    return SymInt(v)


# ---------------------------------------------------------------------------
# cute.math
# ---------------------------------------------------------------------------
class _Math:
    @staticmethod
    def rsqrt(x, loc=None, ip=None, **kw):
        e = _emitter()
        v = x.ssa if isinstance(x, TypedScalar) else x
        r = e.ssa("f32",
                  'nvvm.inline_ptx "rsqrt.approx.f32 $0, $1;" '
                  f'ro ({v.name} : f32) -> f32')
        return _dt.Float32(r)

    @staticmethod
    def exp(x, loc=None, ip=None, **kw):
        e = _emitter()
        v = x.ssa if isinstance(x, TypedScalar) else x
        r = e.ssa("f32",
                  'nvvm.inline_ptx "ex2.approx.f32 $0, $1;" '
                  f'ro ({v.name} : f32) -> f32')
        return _dt.Float32(r)


math = _Math()
