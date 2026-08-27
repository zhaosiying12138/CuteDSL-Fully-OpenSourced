"""Remaining flashinfer-bridge names routed through cute.__getattr__:
autovec_copy + make_rmem_tensor (fragment-level), re-exporting the rest."""
from __future__ import annotations

from cutlass.cute._fi_ext import (Pointer, TensorSSA, ReductionOp,  # noqa: F401
                                  ceil_div, sym_int)


def autovec_copy(src, dst, **kw):
    """cute.autovec_copy(sX_partition, rX_fragment): elementwise smem->rmem.

    src: TVPartView (per-thread units) or smem tensor view; dst: Fragment.
    """
    from cutlass._bridge_helpers import _emitter
    e = _emitter()
    from cutlass.cute._fi_copy import TVPartView

    if isinstance(src, TVPartView):
        T, R, V, B = src._geom
        arr = src.tensor.base
        if hasattr(arr, "ptr"):
            ptr = arr.ptr
        else:
            ptr = arr
        for j in range(V * B):
            row, col = src.coord(j)

            def _c(x):
                return x if hasattr(x, "name") else e.ssa(
                    "index", f"arith.constant {int(x)} : index")

            C_tile = src.tc.tiler[1]
            off = e.idx_binop(
                "arith.addi",
                e.idx_binop("arith.muli", _c(row), _c(int(C_tile))),
                _c(col))
            off64 = e.ssa("i64", f"arith.index_cast {off.name} : index to i64")
            p = e.ssa("!llvm.ptr<3>",
                      f"llvm.getelementptr {ptr.name}[{off64.name}] : "
                      "(!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16")
            v = e.ssa("f16", f"llvm.load {p.name} : !llvm.ptr<3> -> f16")
            dst[j] = v
        return
    # fallback: linear elementwise
    raise NotImplementedError("autovec_copy over this source type")


def make_rmem_tensor(layout, dtype, *a, **kw):
    """cute.make_rmem_tensor(layout, Boolean) etc. — a register fragment."""
    from self_cutedsl.frontend.kernel_objects import Fragment
    from self_cutedsl.frontend.cute_objects import Layout

    def _flat(x):
        out = []
        for e in x:
            if isinstance(e, (tuple, list)):
                out.extend(_flat(e))
            else:
                out.append(int(e))
        return out

    if isinstance(layout, Layout):
        shape = _flat(layout.shape)
    else:
        shape = _flat(layout)
    n = 1
    for s in shape:
        n *= s
    name = getattr(dtype, "name", "f32")
    from self_cutedsl.frontend import builtins as _b
    if name in ("f16", "Float16"):
        elem = _b.F16
    else:
        elem = _b.F32
    return Fragment(n, elem, "rmem")
