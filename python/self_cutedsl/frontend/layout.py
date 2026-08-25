"""layout.py — host-side CuTe layout meta-objects bound to the public cute dialect.

A CuteLayout serializes to MLIR (cute.make_shape/make_stride/make_layout),
which `cute-opt -cute-fold-static` folds; our Python reference evaluator
provides the property-test oracle (size/cosize/eval must agree with the
folded cute.static results).
"""
from __future__ import annotations

from typing import Sequence


def _render(t) -> str:
    if isinstance(t, (tuple, list)):
        return "(" + ",".join(_render(x) for x in t) + ")"
    return str(int(t))


class CuteLayout:
    def __init__(self, shape, stride=None):
        self.shape = _tuplify(shape)
        if stride is None:
            stride = _row_major_stride(self.shape)
        self.stride = _tuplify(stride)

    # ------------------------------------------------------------- MLIR text
    def _decl(self, prefix: str) -> list[str]:
        s, d = _render(self.shape), _render(self.stride)
        lines = [
            f"  %{prefix}_shape = cute.make_shape () : () -> !cute.shape<\"{s}\">",
            f"  %{prefix}_stride = cute.make_stride () : () -> !cute.stride<\"{d}\">",
            f"  %{prefix}_layout = cute.make_layout (%{prefix}_shape, %{prefix}_stride)"
            f" : (!cute.shape<\"{s}\">, !cute.stride<\"{d}\">) -> !cute.layout<\"{s}:{d}\">",
        ]
        return lines

    @property
    def layout_type(self) -> str:
        return f"!cute.layout<\"{_render(self.shape)}:{_render(self.stride)}\">"

    # ------------------------------------------------------- reference eval
    @property
    def size(self) -> int:
        return _prod(self.shape)

    @property
    def cosize(self) -> int:
        idxs = [self.eval(_unravel(i, self.shape)) for i in range(self.size)]
        return (max(idxs) + 1) if idxs else 1

    def eval(self, coord) -> int:
        # hierarchical strides flatten losslessly for linear combination
        return _eval_rec(_flatten_tuple(coord), _flatten_tuple(self.stride))


# ---------------------------------------------------------------- helpers
def _tuplify(x):
    if isinstance(x, (tuple, list)):
        return tuple(_tuplify(i) for i in x)
    return int(x)


def _prod(t) -> int:
    if isinstance(t, tuple):
        r = 1
        for x in t:
            r *= _prod(x)
        return r
    return int(t)


def _row_major_stride(shape):
    flat = _flatten(shape)
    strides, acc = [], 1
    for d in reversed(flat):
        strides.append(acc)
        acc *= d
    return tuple(reversed(strides)) if strides else ()


def _flatten(t):
    out = []

    def rec(x):
        if isinstance(x, tuple):
            for i in x:
                rec(i)
        else:
            out.append(int(x))

    rec(t)
    return out


def _unravel(idx: int, shape) -> tuple:
    flat = _flatten(shape)
    coords = []
    for d in reversed(flat):
        coords.append(idx % d)
        idx //= d
    return tuple(reversed(coords))


def _eval_rec(coord, stride) -> int:
    c = coord if isinstance(coord, (tuple, list)) else (coord,)
    s = stride if isinstance(stride, (tuple, list)) else (stride,)
    if len(c) != len(s):
        raise ValueError(f"coord/stride rank mismatch {c} vs {s}")
    return sum(int(ci) * int(si) for ci, si in zip(c, s))


def _flatten_tuple(t):
    out = []

    def rec(x):
        if isinstance(x, (tuple, list)):
            for i in x:
                rec(i)
        else:
            out.append(int(x))

    rec(t)
    return tuple(out)
