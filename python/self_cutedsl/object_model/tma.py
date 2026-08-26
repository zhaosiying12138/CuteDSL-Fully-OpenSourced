"""object_model/tma.py — generalized TMA atom/partition over the builtins.

S3 changes vs the M6 builtins path:
  * tma_issue accepts FULLY DYNAMIC box coordinates (list of SSA i32),
    removing the hardcoded second-coordinate-0 limitation;
  * prefetch.tensormap via nvvm.inline_ptx;
  * multicast via the cp.async.bulk.tensor multicast operand (mask SSA).
"""
from __future__ import annotations

from ..frontend import builtins as _b


def tma_issue(smem_window, tma_desc_ssa, bar_ssa, coords):
    """cp.async.bulk.tensor G2S with dynamic coords (inner-first)."""
    e = _b._emitter()
    smem_ptr = getattr(smem_window, "ptr", smem_window)
    off = getattr(smem_window, "stage_offset", None)
    if off is not None:
        elem = getattr(smem_window, "elem", None)
        ety = "f16" if getattr(elem, "name", "").lower() in ("f16", "float16") else "f32"
        smem_ptr = e.gep_smem(smem_ptr, off, ety)
    e.tma_load(smem_ptr, tma_desc_ssa, bar_ssa, list(coords))


def tma_issue_multicast(smem_window, tma_desc_ssa, bar_ssa, coords, mask_ssa):
    """G2S multicast: mask = 16-bit CTA rank mask (SSA i16)."""
    e = _b._emitter()
    smem_ptr = getattr(smem_window, "ptr", smem_window)
    cs = []
    for c in coords:
        cs.append(c if getattr(c, "type", "") == "i32"
                  else e.ssa("i32", f"arith.index_cast {c.name} : {c.type} to i32")
                  if not isinstance(c, int) else
                  e.ssa("i32", f"arith.constant {int(c)} : i32"))
    ops = ", ".join(x.name for x in cs)
    # multicast via inline PTX adapter (nvvm op lacks the mask operand)
    e.raw(
        f'nvvm.inline_ptx "cp.async.bulk.tensor.2d.shared::cluster.global.tile'
        f'.mbarrier::complete_tx::bytes.multicast::cluster '
        f'[$0], [$1, {{{ops}}}], [$2], $3;" '
        f'ro ({smem_ptr.name}, {tma_desc_ssa.name}, {bar_ssa.name} : '
        f'!llvm.ptr<3>, !llvm.ptr, !llvm.ptr<3>) rw ({mask_ssa.name} : i16)')


def tma_store_issue(tma_desc_ssa, smem_window, coords):
    """S2G with dynamic coords + bulk-group commit/wait."""
    e = _b._emitter()
    smem_ptr = getattr(smem_window, "ptr", smem_window)
    e.fence_proxy_async_shared()
    e.tma_store(tma_desc_ssa, smem_ptr, list(coords))


def prefetch_tensormap(tma_desc_ssa):
    """prefetch.tensormap 2D/3D/4D/5D — dimension from the descriptor is
    not in the type text, so emit via inline PTX with the generic form."""
    e = _b._emitter()
    e.raw(f'nvvm.inline_ptx "prefetch.tensormap [$0];" '
          f'ro ({tma_desc_ssa.name} : !llvm.ptr)')
