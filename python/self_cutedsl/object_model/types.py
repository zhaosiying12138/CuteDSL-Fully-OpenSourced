"""types.py — CuTe object-model types with textual !cute<...> serialization.

Copyright (c) 2026 CuTeDSL-Fully-OpenSourced contributors
Serialization semantics follow the BSD-3 cutlass_compiler cute dialect
(text grammar observed in third_party/cutlass/cutlass_compiler tests).

Every meta object carries a *textual* MLIR type plus the values needed to
build it:
  * fully static  -> emits `cute.static : !cute.T<"literal">` (one op)
  * dynamic leaf  -> emits cute.make_* op(s) whose dynamic operands are
    SSA references captured by the emitter scope.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _render(t) -> str:
    """Render a (possibly nested) int-or-None tuple as cute literal text."""
    if isinstance(t, (tuple, list)):
        return "(" + ",".join(_render(x) for x in t) + ")"
    if t is None or (isinstance(t, DynLeaf)):
        return "?"
    return str(int(t))


class DynLeaf:
    """A runtime dynamic leaf — binds one SSA value into a type slot."""

    def __init__(self, ssa):
        self.ssa = ssa            # emitter SSA (i32) or None (structure-only)
        self.name = ssa.name if ssa is not None and hasattr(ssa, "name") else "?"

    def __repr__(self):
        return f"?{self.name}"


def _flatten_leaves(t, out):
    if isinstance(t, (tuple, list)):
        for x in t:
            _flatten_leaves(x, out)
    else:
        out.append(t)


def dyn_leaves(t) -> list:
    out: list = []
    _flatten_leaves(t, out)
    return [x for x in out if isinstance(x, DynLeaf)]


def is_static(t) -> bool:
    return not dyn_leaves(t)


@dataclass
class IntTuple:
    value: tuple                 # nested int | DynLeaf | (recursive)

    @property
    def text(self) -> str:
        return f'!cute.int_tuple<"{_render(self.value)}">'

    def py(self):
        return tuple(x.py() if isinstance(x, IntTuple) else x for x in self.value)


@dataclass
class Shape:
    value: tuple

    @property
    def text(self) -> str:
        return f'!cute.shape<"{_render(self.value)}">'

    def py(self):
        return tuple(x.py() if isinstance(x, Shape) else x for x in self.value)


@dataclass
class Stride:
    value: tuple

    @property
    def text(self) -> str:
        return f'!cute.stride<"{_render(self.value)}">'

    def py(self):
        return tuple(x.py() if isinstance(x, Stride) else x for x in self.value)


@dataclass
class Coord:
    value: tuple                 # int | DynLeaf | None (underscore)

    @property
    def text(self) -> str:
        return f'!cute.coord<"{_render(self.value)}">'

    def py(self):
        return tuple(x if not isinstance(x, (Coord, Shape, Stride)) else x.py()
                     for x in self.value)


@dataclass
class Layout:
    shape: tuple                 # nested ints / DynLeaf
    stride: tuple

    @property
    def text(self) -> str:
        return f'!cute.layout<"{_render(self.shape)}:{_render(self.stride)}">'

    def py(self):
        return (self.shape, self.stride)

    # -- algebra helpers on the python static view (all-static only) --
    @staticmethod
    def prod(t) -> int:
        r = 1
        for x in t:
            r *= Layout.prod(x) if isinstance(x, tuple) else int(x)
        return r

    @property
    def size(self) -> int:
        return Layout.prod(self.shape)


@dataclass
class Tile:
    layout: Layout

    @property
    def text(self) -> str:
        return f'!cute.tile<"[{_render(self.layout.shape)}:{_render(self.layout.stride)}]">'


@dataclass
class Swizzle:
    """Swizzle<B, M, S> textual S<B,M,S>."""

    b: int
    m: int
    s: int

    @property
    def text(self) -> str:
        return f'!cute.swizzle<"S<{self.b},{self.m},{self.s}>">'


@dataclass
class ComposedLayout:
    inner: object                # Swizzle | Layout | Tile
    offset: IntTuple
    outer: Layout

    @property
    def text(self) -> str:
        inner_txt = (self.inner.text if isinstance(self.inner, Swizzle)
                     else self.inner.text)  # layout/tile render themselves
        # composed literal: `<inner> o <offset> o <outer>` with layout
        # prints as shape:stride
        return (f'!cute.composed_layout<"{_inner_lit(self.inner)} o '
                f'{_render(self.offset.value)} o {_cl_outer_lit(self.outer)}">')


def _inner_lit(inner) -> str:
    if isinstance(inner, Swizzle):
        return f"S<{inner.b},{inner.m},{inner.s}>"
    if isinstance(inner, Layout):
        return f"{_render(inner.shape)}:{_render(inner.stride)}"
    if isinstance(inner, Tile):
        return f"[{_render(inner.layout.shape)}:{_render(inner.layout.stride)}]"
    raise TypeError(inner)


def _cl_outer_lit(outer: Layout) -> str:
    return f"{_render(outer.shape)}:{_render(outer.stride)}"
