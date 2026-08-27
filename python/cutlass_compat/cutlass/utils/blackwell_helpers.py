"""cutlass.utils.blackwell_helpers — SM120 block-scaled host helpers.

get_permutation_mnk: warp-tile extent for the blockscaled tiled_mma
(the atom grid is (4,2,1) warps with m16n8k64 fp4 / m16n8k32 fp8 atoms).
partition_fragment_SF*: per-thread SF register views (host descriptors
lowered at the copy site). get_layoutSF{A,B}_TV: thread-value layouts.
"""
from cutlass.cute import Layout, make_layout


class _PermutationMNK(tuple):
    def __new__(cls, values, cta_tile_shape_mnk):
        result = super().__new__(cls, values)
        result.cta_tile_shape_mnk = tuple(
            int(value) for value in cta_tile_shape_mnk)
        return result


def get_permutation_mnk(tile_shape_mnk, sf_vec_size, use_mxf8f6f4=False):
    if use_mxf8f6f4:
        # fp8-class atoms m16n8k32: warp tile = (4*16, 2*8*4, 32)
        return _PermutationMNK(
            (4 * 16, 2 * 8 * 4, 32), tile_shape_mnk)
    # fp4-class atoms m16n8k64: warp tile = (4*16, 2*8*2, 64)
    return _PermutationMNK((4 * 16, 2 * 8 * 2, 64), tile_shape_mnk)


def get_layoutSFA_TV(tiled_mma):
    """Thread-value layout for SFA fragments (host descriptor); the operand
    tag travels on the layout so the copy-site lowering knows A vs B."""
    lay = make_layout((4, 2, 1))
    lay._sf_operand = "A" if not __import__("os").environ.get("DG_NO_TVTAG") else None
    lay._tiled_mma = tiled_mma
    return lay


def get_layoutSFB_TV(tiled_mma):
    lay = make_layout((4, 2, 1))
    lay._sf_operand = "B" if not __import__("os").environ.get("DG_NO_TVTAG") else None
    lay._tiled_mma = tiled_mma
    return lay


def partition_fragment_SFA(sSFA_view, thr_mma, tidx):
    """Per-thread SFA register view over the staged SMEM tile: a
    descriptor carrying (smem tile, tidx, which='A') for the copy-site
    lowering."""
    return _SFragView(sSFA_view, thr_mma, tidx, "A")


def partition_fragment_SFB(sSFB_view, thr_mma, tidx):
    return _SFragView(sSFB_view, thr_mma, tidx, "B")


class _SFragView:
    def __init__(self, smem_view, thr_mma, tidx, which):
        self.smem_view = smem_view
        self.thr_mma = thr_mma
        self.tidx = tidx
        self.which = which

    @property
    def shape(self):
        # ((atom), rest...) descriptor shape for rank/group_modes
        return (4, 2, 1)

    @property
    def rank(self):
        return len(self.shape)

    def __getitem__(self, idx):
        idx = idx if isinstance(idx, tuple) else (idx,)
        slots = [c for c in idx if c is not None]
        v = _SFragView(self.smem_view, self.thr_mma, self.tidx, self.which)
        v.k = slots[-1] if slots else 0
        return v


def sm90_get_smem_store_op(*args, **kw):
    return ("stmatrix", "f16")


def sm120_get_smem_store_op(*args, **kw):
    return ("stmatrix", "f16")
