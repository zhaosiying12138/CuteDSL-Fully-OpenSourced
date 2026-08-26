"""cutlass.utils.blockscaled_layout — SM120 block-scaled SF layout helpers.

The scale-factor tensors live in the canonical CUTLASS blocked layout
M(32x4xrest_m) x K(4xrest_k) x L (the TMA-friendly atom_m=(32,4), atom_k=4
form produced by the examples' create_scale_factor_tensor). These helpers
tile that layout to the kernel's tile shapes; all host-meta, mirroring the
dense_gemm layout helpers.
"""
from cutlass.cute import Layout, make_layout


def _tile_shape_sf(tile_shape_mnk, sf_vec_size):
    """SF elements per (m, k) tile: (tile_m, tile_k / sf_vec)."""
    tm, tn, tk = tile_shape_mnk
    return (tm, tk // sf_vec_size)


def tile_atom_to_shape_SF(sf_atom_layout, shape_mnkl, sf_vec_size,
                          tile_shape_mnk):
    """Tile the full-SF-tensor layout by the per-tile SF atom.

    sf_atom_layout: the per-tile SF layout (m, k_sf) from
    sm120_make_smem_layout_{sfa,sfb} minus staging.
    Returns a staged-style host layout: ((atom), (rest_m, rest_k, l))."""
    from cutlass.cute import Layout as L

    m, n, k, l = (int(x) for x in shape_mnkl)
    atom_m, atom_k = _tile_shape_sf(tile_shape_mnk, sf_vec_size)
    rest_m = -(-m // atom_m)
    rest_k = -(-k // (atom_k * sf_vec_size))
    atom_shape = tuple(int(d) for d in
                       (sf_atom_layout.shape
                        if isinstance(sf_atom_layout, Layout) else atom_m,))
    # the atom from the smem layout is (m, k_sf) nested or flat
    flat = []
    for d in sf_atom_layout.shape:
        if isinstance(d, tuple):
            flat.extend(d)
        else:
            flat.append(int(d))
    atom_lay = L(tuple(flat), tuple(range(len(flat) - 1, -1, -1)))
    return L(((atom_lay.shape), (rest_m, rest_k, l)),
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
