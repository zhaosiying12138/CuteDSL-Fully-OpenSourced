"""cutlass.cute.nvgpu.cpasync — TMA atom object model (compat)."""
from __future__ import annotations


class CopyBulkTensorTileG2SOp:
    pass


class CopyBulkTensorTileG2SMulticastOp:
    pass


class CopyBulkTensorTileS2GOp:
    pass


class CopyUniversalOp:
    pass


def make_tiled_tma_atom(op, gmem_tensor, smem_layout_or_tensor, cta_layout=None, **kw):
    from self_cutedsl.frontend import builtins

    return builtins.make_tiled_tma_atom(op, gmem_tensor,
                                        smem_layout_or_tensor, cta_layout)


def tma_partition(atom, cta_coord, cta_layout, smem_grouped, gmem_grouped):
    from self_cutedsl.frontend import builtins

    return builtins.tma_partition(atom, cta_coord, cta_layout,
                                  smem_grouped, gmem_grouped)


def prefetch_descriptor(tma_tensor):
    pass  # perf hint only; no-op in first self-stack version
