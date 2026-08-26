"""cutlass.utils.hopper_helpers — SM90-era smem layout helpers (compat).

Re-exported SM120-capable implementations from cutlass.utils; the
flagship SM120 kernels import this module for the layout helpers only.
"""
from __future__ import annotations

from . import (compute_tile_shape_or_override, make_smem_layout_a,
               make_smem_layout_b, make_smem_layout_epi,
               sm90_get_smem_store_op, ComposedLayoutStaged)

__all__ = ["compute_tile_shape_or_override", "make_smem_layout_a",
           "make_smem_layout_b", "make_smem_layout_epi",
           "sm90_get_smem_store_op", "ComposedLayoutStaged"]


def sm90_get_smem_store_op(c_layout, elem_ty_d=None, elem_ty_acc=None, **kw):
    """Epilogue r2s op marker (StMatrix-class); lowering keys off the mma
    trait table, so a plain marker suffices."""
    return ("stmatrix", getattr(elem_ty_d, "name", None) or "f16")


def get_smem_layout_atom(layout_enum, dtype=None, major_mode_size=64):
    """Canonical SMEM layout atom descriptor for the operand: returns the
    (m, k) tile extents per atom (K-major), host meta."""
    from cutlass.cute import make_layout

    k = int(major_mode_size) or 64
    return make_layout((64 if dtype is None or
                        getattr(dtype, "width", 16) >= 8 else 64, k))
