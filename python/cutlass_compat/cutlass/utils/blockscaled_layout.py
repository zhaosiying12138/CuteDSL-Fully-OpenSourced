"""cutlass.utils.blockscaled_layout — SM120 block-scaled SF layout helpers.

The scale-factor tensors live in the canonical CUTLASS blocked layout
M(32x4xrest_m) x K(4xrest_k) x L (the TMA-friendly atom_m=(32,4), atom_k=4
form produced by the examples' create_scale_factor_tensor). These helpers
tile that layout to the kernel's tile shapes; all host-meta, mirroring the
dense_gemm layout helpers.
"""
from cutlass.cute import make_layout


def _tile_shape_sf(tile_shape_mnk, sf_vec_size):
    """SF elements per (m, k) tile: (tile_m, tile_k / sf_vec)."""
    tm, tn, tk = tile_shape_mnk
    return (tm, tk // sf_vec_size)


def tile_atom_to_shape_SF(tensor_or_shape, sf_vec_size,
                          tile_shape_mnk=(128, 128, 128)):
    """SF tensor layout tiled by the canonical 32x4 x K4 atom:
    the input is the SF tensor (m, k/sf_vec, l meta) or its shape; returns
    ((atom_m, atom_k), (rest_m, rest_k, l)) host layout. The gmem SF data
    already lives in the M32x4xK4 blocked order, so the atom maps 1:1."""
    from self_cutedsl.frontend.cute_objects import Layout as L

    if hasattr(tensor_or_shape, "shape"):
        shape = tuple(int(d) for d in tensor_or_shape.shape)
    else:
        shape = tuple(int(d) for d in tensor_or_shape)
    # the SF tensor arrives as (m, k_sf, l) logical elements
    if len(shape) == 3:
        m, k_sf, l = shape
    elif len(shape) == 2:
        m, k_sf, l = shape[0], shape[1], 1
    else:
        raise NotImplementedError(f"SF shape {shape}")
    rest_m = -(-m // 128)
    rest_k = -(-k_sf // 4)
    return L(((32, 4, 4), (rest_m, rest_k, l)),
             ((0, 0, 0), (0, 0, 0)))


def sm120_make_smem_layout_sfa(tiled_mma, tile_shape_mnk, sf_vec_size,
                               ab_stage):
    """Per-stage SMEM layout for SFA: (tile_m, tile_k/sf_vec) elements."""
    from cutlass.utils import ComposedLayoutStaged

    tm = int(tile_shape_mnk[0])
    tk_sf = int(tile_shape_mnk[2]) // int(sf_vec_size)
    outer = make_layout((tm, tk_sf))
    return ComposedLayoutStaged(outer, ab_stage)


def sm120_make_smem_layout_sfb(tiled_mma, tile_shape_mnk, sf_vec_size,
                               ab_stage):
    """Per-stage SMEM layout for SFB: (tile_n, tile_k/sf_vec)."""
    from cutlass.utils import ComposedLayoutStaged

    tn = int(tile_shape_mnk[1])
    tk_sf = int(tile_shape_mnk[2]) // int(sf_vec_size)
    outer = make_layout((tn, tk_sf))
    return ComposedLayoutStaged(outer, ab_stage)
