"""cutlass.cute.nvgpu.warpgroup — SM120 blockscaled layout helpers."""


def make_smem_layout_atom(*args, **kw):
    """Per-stage SMEM layout atom for A/B/C operands (host meta): the
    (tile_m, tile_k) K-major layout; staged composition happens at the
    caller via tile_to_shape-style staging."""
    from cutlass.cute import make_layout

    # call forms: (dtype, layout_enum) / (dtype) — return a canonical
    # K-major atom; the exact swizzle choice rides the composed path.
    return make_layout((64, 64))
