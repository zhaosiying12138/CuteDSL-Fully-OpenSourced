"""mma_atoms.py — per-atom closed-form trait tables for TiledMma partitioning.

Copyright (c) 2026 CuTeDSL-Fully-OpenSourced contributors
Trait values calibrated against the verified handwritten kernels in
tests/ (m16n8k16 offsets validated golden since M4).

The open cute dialect does NOT ship TiledMma ops (official partitioning
lives in the closed cute_nvgpu). This module is the sanctioned extension
point: each atom contributes a table of per-thread fragment layouts;
partition_A/B/C derive their index arithmetic GENERALLY through
algebra.py ops — never per-kernel hand-written offsets.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MmaAtomTraits:
    """Closed-form per-thread fragment traits for one warp MMA atom."""

    m: int
    n: int
    k: int
    threads: int = 32
    frag_a_shape: tuple = ()      # per-thread extents
    frag_b_shape: tuple = ()
    frag_c_shape: tuple = ()
    a_regs: int = 0
    b_regs: int = 0
    c_regs: int = 0
    thr_layout: tuple = ()        # threads tile (atom_m x atom_n)
    val_layout: tuple = ()


def _m16n8k16_traits() -> MmaAtomTraits:
    # PTX mma.sync.aligned.m16n8k16 f16 lane ownership:
    #  A rows {l//4, l//4+8}, cols {2*(l%4), +1, +8, +9}  (8 f16 = 4 regs)
    #  B k-rows {2*(l%4), +1, +8, +9}, n-col {l//4}       (4 f16 = 2 regs)
    #  C rows {l//4, l//4+8}, cols {2*(l%4), +1}          (4 f32 = 4 regs)
    return MmaAtomTraits(
        m=16, n=8, k=16,
        frag_a_shape=(2, 2, 2),
        frag_b_shape=(2, 2),
        frag_c_shape=(2, 2),
        a_regs=4, b_regs=2, c_regs=4,
        thr_layout=(8, 4),
        val_layout=(2, 2),
    )


ATOM_TABLE = {
    ("m16n8k16", "f16"): _m16n8k16_traits(),
}


def register_atom(name: str, dtype_key: str, traits: MmaAtomTraits) -> None:
    """Extension point: a new atom is a table row; algebra never changes."""
    ATOM_TABLE[(name, dtype_key)] = traits


def a_fragment_coords(traits: MmaAtomTraits, lane: int):
    """(row, col) pairs of A elements owned by `lane`."""
    if (traits.m, traits.n, traits.k) != (16, 8, 16):
        raise NotImplementedError("only m16n8k16 calibrated")
    group, tig = lane // 4, lane % 4
    return [(r, c)
            for r in (group, group + 8)
            for c in (2 * tig, 2 * tig + 1, 2 * tig + 8, 2 * tig + 9)]


def b_fragment_coords(traits: MmaAtomTraits, lane: int):
    """(k-row, n-col) of B elements owned by `lane`."""
    if (traits.m, traits.n, traits.k) != (16, 8, 16):
        raise NotImplementedError
    group, tig = lane // 4, lane % 4
    return [(c, group)
            for c in (2 * tig, 2 * tig + 1, 2 * tig + 8, 2 * tig + 9)]


def c_fragment_coords(traits: MmaAtomTraits, lane: int):
    """(row, col) of C elements owned by `lane`."""
    if (traits.m, traits.n, traits.k) != (16, 8, 16):
        raise NotImplementedError
    group, tig = lane // 4, lane % 4
    return [(group, 2 * tig), (group, 2 * tig + 1),
            (group + 8, 2 * tig), (group + 8, 2 * tig + 1)]
