"""cutegen_binding.py — in-process oracle over the BSD cutegen algebra.

The vendored cutlass_compiler's cute dialect uses cutegen (header-only,
BSD-3) as its layout/shape type engine — the same library the closed-source
official DSL links. The nanobind binding (tools/cutegen_oracle/, built into
build-oracle/_cutegen_oracle.so by tools/cutegen_oracle/build.sh) exposes
that algebra in-process, replacing the bootstrap "ask the verifier binary
via stderr diagnostics" path with a direct call.

Semantics and limits (documented in the report):
  * text-in / text-out in the cute layout grammar ('?' = dynamic);
  * the default dynamic_traits_t is stateless: '?' leaves are ANONYMOUS —
    an identity-carrying dynamic requires an emission backend mirroring
    mlir_dynamic.hpp (~600 lines of specializations), which the closed
    .so implements and we do not (our dynamic identity lives in the
    scalar-ABI layer instead);
  * every result is cached by (op, operands); misses raise loudly.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_SO_DIR = ROOT / "build-oracle"

_mod = None
_load_error = None
_cache: dict = {}


def _load():
    global _mod, _load_error
    if _mod is not None or _load_error is not None:
        return _mod
    try:
        if str(_SO_DIR) not in sys.path:
            sys.path.insert(0, str(_SO_DIR))
        import _cutegen_oracle as m  # noqa: N813
        _mod = m
    except Exception as e:  # pragma: no cover - environment-dependent
        _load_error = e
    return _mod


def available() -> bool:
    return _load() is not None


def unavailable_reason() -> str:
    _load()
    if _load_error is None:
        return ""
    return (f"cutegen binding not built ({_load_error}); "
            f"run tools/cutegen_oracle/build.sh")


# type-text <-> layout-text helpers -----------------------------------------
_LAYOUT_TYPE_RE = re.compile(r'^!cute\.layout<"(.*)">$')
_SHAPE_TYPE_RE = re.compile(r'^!cute\.shape<"(.*)">$')
_INTT_TYPE_RE = re.compile(r'^!cute\.int_tuple<"(.*)">$')


def layout_text_of(type_text: str) -> str:
    m = _LAYOUT_TYPE_RE.match(type_text.strip())
    if not m:
        raise ValueError(f"not a !cute.layout type text: {type_text!r}")
    return m.group(1)


def as_layout_type(layout_text: str) -> str:
    return f'!cute.layout<"{layout_text}">'


# oracle ops ----------------------------------------------------------------
def _call(op: str, *args: str) -> str:
    key = (op, args)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    m = _load()
    if m is None:
        raise RuntimeError(unavailable_reason())
    fn = getattr(m, op)
    out = fn(*args)
    _cache[key] = out
    return out


def composition(a_type: str, b_type: str) -> str:
    return as_layout_type(_call(
        "composition", layout_text_of(a_type), layout_text_of(b_type)))


def coalesce(a_type: str) -> str:
    return as_layout_type(_call("coalesce", layout_text_of(a_type)))


def flatten(a_type: str) -> str:
    return as_layout_type(_call("flatten", layout_text_of(a_type)))


def _kind_of(type_text: str) -> str:
    t = type_text.strip()
    if t.startswith("!cute.shape"):
        return "shape"
    if t.startswith("!cute.tile"):
        return "tile"
    if t.startswith("!cute.layout"):
        return "layout"
    # bare text defaults to the layout overload
    return "layout"


def zipped_divide(a_type: str, t_type: str) -> str:
    return as_layout_type(_call(
        "zipped_divide", layout_text_of(a_type),
        _strip_type(t_type), _kind_of(t_type)))


def logical_divide(a_type: str, t_type: str) -> str:
    return as_layout_type(_call(
        "logical_divide", layout_text_of(a_type),
        _strip_type(t_type), _kind_of(t_type)))


def slice_(crd_type: str, lay_type: str) -> str:
    return as_layout_type(_call(
        "slice", _strip_type(crd_type), layout_text_of(lay_type)))


def _strip_type(type_text: str) -> str:
    t = type_text.strip()
    for rx in (_LAYOUT_TYPE_RE, _SHAPE_TYPE_RE, _INTT_TYPE_RE):
        m = rx.match(t)
        if m:
            return m.group(1)
    return t


# group_modes is a purely structural mode regroup (no algebra); its result
# metas stay caller-side and are cross-checked against the verifier in the
# differential harness rather than half-reimplemented here.
def group_modes(a_type: str, i: int, j: int) -> str:
    raise NotImplementedError(
        "group_modes result metas stay caller-side (structural regroup, "
        "no cutegen algebra involved); see the differential harness")


def count_dynamics(a_type: str) -> int:
    return int(_call("count_dynamics", layout_text_of(a_type)))


def stats() -> tuple:
    m = _load()
    if m is None:
        return (0, 0.0)
    return m.stats()
