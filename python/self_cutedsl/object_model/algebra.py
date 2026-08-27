"""algebra.py — CuTe layout algebra: text-emission layer over the cute dialect.

Copyright (c) 2026 CuTeDSL-Fully-OpenSourced contributors
Algebra semantics are owned by the C++ passes in cutlass-compiler
(--cute-fold-static / --cute-expand-ops / --cute-to-base). This layer:

  * fully static values   -> python constant folding only where needed
    for trace-time host decisions (size/cosize of static layouts);
  * anything with a dynamic leaf -> emits the matching cute.* op text
    and RETURNS a typed meta handle carrying the op's result type, so
    composition chains serialize without re-implementing semantics.

Emitter interface: an object with .ssa(...) / .raw(...) — the existing
KernelEmitter satisfies it. Dynamic leaves are `DynLeaf(ssa)`.
"""
from __future__ import annotations

from .types import (DynLeaf, IntTuple, Shape, Stride, Coord, Layout, Tile,
                    Swizzle, ComposedLayout, dyn_leaves, is_static, _render)


class CuteOpValue:
    """Result of an emitted cute op: knows its textual type + defining SSA."""

    def __init__(self, ssa, ctype):
        self.ssa = ssa          # emitter SSA of the op result
        self.ctype = ctype      # e.g. Layout/Shape/... meta (type-level info)

    @property
    def name(self):
        return self.ssa.name


def _static_op(e, lit: str):
    return e.ssa(lit, f'cute.static : {lit}')


# ------------------------------------------------------------------ make_*
def make_int_tuple(e, value):
    """IntTuple from possibly-dynamic leaves."""
    if is_static(value):
        t = IntTuple(value)
        return CuteOpValue(_static_op(e, t.text), t)
    leaves = dyn_leaves(value)
    opnds = ", ".join(x.name for x in leaves)
    t = IntTuple(value)
    res = e.ssa(t.text, f"cute.make_int_tuple ({opnds}) "
                        f": ({', '.join(['i32'] * len(leaves))}) -> {t.text}")
    return CuteOpValue(res, t)


def make_shape(e, value):
    if is_static(value):
        t = Shape(value)
        return CuteOpValue(_static_op(e, t.text), t)
    leaves = dyn_leaves(value)
    opnds = ", ".join(x.name for x in leaves)
    t = Shape(value)
    res = e.ssa(t.text, f"cute.make_shape ({opnds}) "
                        f": ({', '.join(['i32'] * len(leaves))}) -> {t.text}")
    return CuteOpValue(res, t)


def make_stride(e, value):
    if is_static(value):
        t = Stride(value)
        return CuteOpValue(_static_op(e, t.text), t)
    leaves = dyn_leaves(value)
    opnds = ", ".join(x.name for x in leaves)
    t = Stride(value)
    res = e.ssa(t.text, f"cute.make_stride ({opnds}) "
                        f": ({', '.join(['i32'] * len(leaves))}) -> {t.text}")
    return CuteOpValue(res, t)


def make_layout(e, shape_v, stride_v):
    """shape_v/stride_v: python tuples OR CuteOpValue (Shape/Stride)."""
    sh_t = shape_v if isinstance(shape_v, CuteOpValue) else make_shape(e, shape_v)
    st_t = stride_v if isinstance(stride_v, CuteOpValue) else make_stride(e, stride_v)
    # combined type
    sh_meta = sh_t.ctype.value
    st_meta = st_t.ctype.value
    lay = Layout(sh_meta, st_meta)
    res = e.ssa(lay.text, f"cute.make_layout ({sh_t.name}, {st_t.name}) "
                          f": ({sh_t.ctype.text}, {st_t.ctype.text}) -> {lay.text}")
    return CuteOpValue(res, lay)


def make_coord(e, value):
    value = tuple(x if x is not None else None for x in
                  (value if isinstance(value, (tuple, list)) else (value,)))
    if is_static([x for x in value if x is not None]):
        t = Coord(value)
        return CuteOpValue(_static_op(e, t.text), t)
    leaves = [x for x in dyn_leaves(value) if x is not None]
    opnds = ", ".join(x.name for x in leaves)
    t = Coord(value)
    res = e.ssa(t.text, f"cute.make_coord ({opnds}) "
                        f": ({', '.join(['i32'] * len(leaves))}) -> {t.text}")
    return CuteOpValue(res, t)


# ------------------------------------------------------------- accessors
def _unary(e, op, val: CuteOpValue, out_meta, out_text=None):
    txt = out_text or out_meta.text
    res = e.ssa(txt, f"cute.{op}({val.name}) : ({val.ctype.text}) -> {txt}")
    return CuteOpValue(res, out_meta)


def get_shape(e, lay: CuteOpValue):
    sh = Shape(lay.ctype.shape)
    return _unary(e, "get_shape", lay, sh)


def get_stride(e, lay: CuteOpValue):
    st = Stride(lay.ctype.stride)
    return _unary(e, "get_stride", lay, st)


def size(e, x, mode=None):
    """cute.size(layout/shape[, mode]) — int_tuple result; static inputs
    get the exact literal type (inference-verified), dynamic get "?"."""
    if isinstance(x, CuteOpValue):
        if is_static(getattr(x.ctype, "shape", getattr(x.ctype, "value", ()))):
            # exact static literal — compute for the type
            try:
                from .types import Layout as _L
                meta = getattr(x.ctype, "shape", None)
                if meta is None and isinstance(x.ctype, Layout):
                    n = x.ctype.size
                elif meta is not None:
                    n = 1
                    for d in _flat_ints(meta):
                        n *= int(d)
                else:
                    n = None
            except Exception:
                n = None
            txt = f'!cute.int_tuple<"{n}">' if n is not None else '!cute.int_tuple<"?">'
        else:
            txt = '!cute.int_tuple<"?">'
        if mode is not None:
            res = e.ssa(txt, f"cute.size({x.name}) {{mode = {int(mode)} : i32}} "
                             f": ({x.ctype.text}) -> {txt}")
        else:
            res = e.ssa(txt, f"cute.size({x.name}) : ({x.ctype.text}) -> {txt}")
        return CuteOpValue(res, IntTuple((DynLeaf(res),)))
    # python-static shortcut for trace-time host decisions
    from .types import Layout as _L
    if isinstance(x, (tuple, list)):
        r = 1
        for d in x:
            r *= int(d)
        return r
    if isinstance(x, Layout):
        return x.size
    return int(x)


# ------------------------------------------------------------- algebra
import os as _os_algebra

from . import cutegen_binding as _oracle


def _oracle_type(op_name, *type_texts):
    """Result type from the in-process cutegen oracle (the same library the
    cute dialect and the closed official DSL use as their type engine;
    results cached inside the binding). Returns None when the binding is
    unavailable — callers then fall back to the verifier-subprocess
    bootstrap path."""
    if _os_algebra.environ.get("SC_ORACLE_OFF"):
        return None
    if not _oracle.available():
        return None
    try:
        return getattr(_oracle, op_name)(*type_texts)
    except Exception:
        return None


def _emit_typed(e, line: str, result_type_text: str):
    res = e.ssa(result_type_text, line)
    return CuteOpValue(res, _meta_from_text(result_type_text))


def _binary(e, op, a: CuteOpValue, b: CuteOpValue, out_meta):
    res = e.ssa(out_meta.text,
                f"cute.{op}({a.name}, {b.name}) "
                f": ({a.ctype.text}, {b.ctype.text}) -> {out_meta.text}")
    return CuteOpValue(res, out_meta)


def composition(e, a: CuteOpValue, b: CuteOpValue, out_meta=None):
    rt = out_meta.text if out_meta is not None else \
        _oracle_type("composition", a.ctype.text, b.ctype.text)
    if rt is not None:
        return _emit_typed(
            e, f"cute.composition({a.name}, {b.name}) "
               f": ({a.ctype.text}, {b.ctype.text}) -> {rt}", rt)
    return emit_with_inference(
        e, lambda r: f"cute.composition({a.name}, {b.name}) "
                     f": ({a.ctype.text}, {b.ctype.text}) -> {r}")


def coalesce(e, lay: CuteOpValue, out_meta=None):
    rt = out_meta.text if out_meta is not None else \
        _oracle_type("coalesce", lay.ctype.text)
    if rt is not None:
        return _emit_typed(
            e, f"cute.coalesce({lay.name}) : ({lay.ctype.text}) -> {rt}", rt)
    return _unary(e, "coalesce", lay, lay.ctype)


def flatten(e, lay: CuteOpValue, out_meta=None):
    rt = out_meta.text if out_meta is not None else \
        _oracle_type("flatten", lay.ctype.text)
    if rt is not None:
        return _emit_typed(
            e, f"cute.flatten({lay.name}) : ({lay.ctype.text}) -> {rt}", rt)
    if out_meta is not None:
        return _unary(e, "flatten", lay, out_meta)
    raise RuntimeError("flatten: no oracle and no out_meta")


def zipped_divide(e, lay: CuteOpValue, tiler, out_meta=None):
    """tiler: CuteOpValue of Shape/Layout/Tile. Result type from the
    in-process cutegen oracle; verifier path as bootstrap fallback."""
    rt = out_meta.text if out_meta is not None else \
        _oracle_type("zipped_divide", lay.ctype.text, tiler.ctype.text)
    if rt is not None:
        return _emit_typed(
            e, f"cute.zipped_divide({lay.name}, {tiler.name}) "
               f": ({lay.ctype.text}, {tiler.ctype.text}) -> {rt}", rt)
    if out_meta is not None:
        return _binary(e, "zipped_divide", lay, tiler, out_meta)
    return emit_with_inference(
        e, lambda r: f"cute.zipped_divide({lay.name}, {tiler.name}) "
                     f": ({lay.ctype.text}, {tiler.ctype.text}) -> {r}")


def logical_divide(e, lay: CuteOpValue, tiler, out_meta=None):
    rt = out_meta.text if out_meta is not None else \
        _oracle_type("logical_divide", lay.ctype.text, tiler.ctype.text)
    if rt is not None:
        return _emit_typed(
            e, f"cute.logical_divide({lay.name}, {tiler.name}) "
               f": ({lay.ctype.text}, {tiler.ctype.text}) -> {rt}", rt)
    if out_meta is not None:
        return _binary(e, "logical_divide", lay, tiler, out_meta)
    return emit_with_inference(
        e, lambda r: f"cute.logical_divide({lay.name}, {tiler.name}) "
                     f": ({lay.ctype.text}, {tiler.ctype.text}) -> {r}")


def slice_(e, lay: CuteOpValue, crd: CuteOpValue, out_meta=None):
    rt = out_meta.text if out_meta is not None else \
        _oracle_type("slice_", crd.ctype.text, lay.ctype.text)
    if rt is not None:
        return _emit_typed(
            e, f"cute.slice({lay.name}, {crd.name}) "
               f": ({lay.ctype.text}, {crd.ctype.text}) -> {rt}", rt)
    if out_meta is not None:
        return _binary(e, "slice", lay, crd, out_meta)
    raise RuntimeError("slice: no oracle and no out_meta")


def group_modes(e, x: CuteOpValue, i: int, j: int, out_meta=None):
    if out_meta is not None:
        res = e.ssa(out_meta.text,
                    f"cute.group_modes<{int(i)}, {int(j)}>({x.name}) "
                    f": ({x.ctype.text}) -> {out_meta.text}")
        return CuteOpValue(res, out_meta)
    return emit_with_inference(
        e, lambda rt: f"cute.group_modes<{int(i)}, {int(j)}>({x.name}) "
                      f": ({x.ctype.text}) -> {rt}")


def layout_eval(e, crd: CuteOpValue, lay: CuteOpValue):
    """Result int_tuple structure is computed by the C++ verifier."""
    return emit_with_inference(
        e, lambda rt: f"cute.layout_eval({crd.name}, {lay.name}) "
                      f": ({crd.ctype.text}, {lay.ctype.text}) -> {rt}",
        placeholder='!cute.int_tuple<"1">')


def get_scalars(e, it: CuteOpValue):
    """Extract the single scalar of an int_tuple (rank-1)."""
    res = e.ssa("i32", f"cute.get_scalars({it.name}) : ({it.ctype.text}) -> i32")
    return res


def _flat_ints(t):
    out = []

    def rec(v):
        if isinstance(v, (tuple, list)):
            for i in v:
                rec(i)
        else:
            out.append(v)

    rec(t)
    return out


# ---------------------------------------------------------------------------
# Type inference via the C++ verifier (the inference IS the C++ algebra).
# ---------------------------------------------------------------------------
import re as _re
import subprocess as _sp

_INFER_RE = _re.compile(r"inferred type\(s\) '([^']+)'")


def _cc_path():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[3]
    return str(root / "build-compiler/tools/cutlass-compiler/cutlass-compiler")


def infer_result_type(op_text: str, timeout: int = 30) -> str | None:
    """Run cutlass-compiler's verifier as a type oracle.

    op_text: a full module containing the op with a DELIBERATELY WRONG
    result type (e.g. '!cute.layout<"0:0">'). The verifier error names
    the inferred type; return it. Returns None if no inference error.
    """
    proc = _sp.run([_cc_path(), "-"], input=op_text, capture_output=True,
                   text=True, timeout=timeout)
    m = _INFER_RE.search(proc.stderr)
    return m.group(1) if m else None


_PARSEABLE_PLACEHOLDER = '!cute.layout<"(1):(1)">'


def emit_with_inference(e, op_line_builder, placeholder=None):
    """Emit an op whose RESULT TYPE is computed by the C++ verifier (the
    inference is the C++ algebra — python never derives shapes).

    op_line_builder(result_type_text) -> the op line (sans SSA 'name =').
    placeholder: parseable wrong-type text of the right KIND (layout vs
    int_tuple); defaults to a layout placeholder.
    Returns a CuteOpValue carrying the verifier-computed textual type.
    """
    ph = placeholder or _PARSEABLE_PLACEHOLDER
    probe = e.ssa(ph, op_line_builder(ph))
    body = "\n".join(e._lines)
    module = ("module {\n  func.func @probe() {\n"
              f"{body}\n    return\n  }}\n}}\n")
    real = infer_result_type(module)
    if real is None:
        # placeholder accepted — result genuinely is (1):(1)
        real = ph
    else:
        e._lines[-1] = e._lines[-1].replace(ph, real, 1)
    probe.ctype = _meta_from_text(real)
    return probe


class RawType:
    """Type handle for non-layout inference results (int_tuple etc.)."""

    def __init__(self, txt: str):
        self._txt = txt

    @property
    def text(self) -> str:
        return self._txt


def _meta_from_text(txt: str):
    """Parse a verifier-inferred textual type back to a meta handle.
    Layouts get structural metas; other kinds get RawType."""
    m = _re.match(r'!cute\.layout<"(.*)">$', txt)
    if not m:
        return RawType(txt)
    sh_s, _, st_s = m.group(1).partition(":")
    return Layout(_parse_shape_str(sh_s), _parse_shape_str(st_s))


def _parse_shape_str(s: str):
    if not s.strip():
        return ()
    # tuple grammar: '(' a ',' b ')' | int | '?'
    pos = 0

    def parse():
        nonlocal pos
        while pos < len(s) and s[pos] == " ":
            pos += 1
        if s[pos] == "(":
            pos += 1
            items = []
            while True:
                items.append(parse())
                while pos < len(s) and s[pos] == " ":
                    pos += 1
                if pos < len(s) and s[pos] == ",":
                    pos += 1
                    continue
                break
            if pos < len(s) and s[pos] == ")":
                pos += 1
            return tuple(items)
        if s[pos] == "?":
            pos += 1
            return DynLeaf(None)
        j = pos
        while j < len(s) and s[j].isdigit():
            j += 1
        v = int(s[pos:j])
        pos = j
        return v

    return parse()
