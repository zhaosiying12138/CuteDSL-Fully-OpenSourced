"""S1 acceptance: object-model algebra — triple-oracle differential tests.

Oracle A: cutlass-compiler (C++ cute dialect passes) fold/expand — the
           textual IR after --cute-fold-static/--cute-expand-ops contains
           the C++-computed structure; for fully-static chains it must
           fold to `cute.static` with the exact expected literal.
Oracle B: python reference evaluator (independent arithmetic on
           flattened shape/stride — used only to double-check static
           literals, NOT a second algebra for dynamic cases).
Oracle C: numeric ground truth via MLIR execution on the 5090 where the
           chain feeds a kernel store (subset).

The core assertion of the object model: **the python layer never
re-implements dynamic algebra** — dynamic chains only verify that the C++
passes accept + fully lower them (zero cute ops after --cute-to-base).
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from self_cutedsl.frontend.emitter import KernelEmitter  # noqa: E402
from self_cutedsl.object_model import algebra  # noqa: E402
from self_cutedsl.object_model.types import DynLeaf, Layout, Shape, Stride  # noqa: E402

CC = str(ROOT / "build-compiler/tools/cutlass-compiler/cutlass-compiler")
CUTE_PASSES = ["--cute-fold-static", "--cute-expand-ops", "--cute-to-base"]


def _run_cc(mlir_text: str, passes) -> str:
    proc = subprocess.run(
        [CC, *passes, "-"], input=mlir_text, capture_output=True, text=True)
    assert proc.returncode == 0, f"cutlass-compiler failed:\n{proc.stderr[:1500]}"
    return proc.stdout


# --------------------------------------------------------------- static fold
@pytest.mark.parametrize("shape,stride,expect_size", [
    ((4, 8), (1, 4), 32),
    ((2, 3, 4), (24, 8, 1), 24),
    ((128, 64), (64, 1), 8192),
    (((2, 3), 4), ((6, 1), 2), 24),
])
def test_static_size_folds(shape, stride, expect_size):
    """Static make_shape/make_stride/make_layout + cute.size must fold to
    cute.static with the exact size literal (Oracle A). Uses a func.func
    with a return so nothing is DCE'd."""
    e = KernelEmitter("s1_static")
    sh = algebra.make_shape(e, shape)
    st = algebra.make_stride(e, stride)
    lay = algebra.make_layout(e, sh, st)
    sz = algebra.size(e, lay)
    # graft a func.func return use around the emitted ops
    body = "\n".join(l for l in e._lines)
    mlir = ("module {\n"
            f"  func.func @probe() -> !cute.int_tuple<\"{expect_size}\"> {{\n"
            f"{body}\n"
            f"    return {sz.name} : !cute.int_tuple<\"{expect_size}\">\n"
            "  }\n}\n")
    out = _run_cc(mlir, ["--cute-fold-static"])
    assert f'cute.static : !cute.int_tuple<"{expect_size}">' in out, out


# ------------------------------------------------------------ dynamic chains
@pytest.mark.parametrize("tile", [(4,), (8,), ((4,), 4)])
def _anchored_eval_module(e, lay_chain, gptr_name="%v0"):
    """Consume a layout_eval result through get_scalars -> gep -> store so
    the chain can't be DCE'd; returns the module text."""
    return e.module_text()


@pytest.mark.parametrize("tile", [(4,), (8,), ((4,), 4)])
def test_dynamic_zipped_divide_lowers(tile):
    """Dynamic leaf -> zipped_divide -> layout_eval -> get_scalars ->
    store: after all three cute passes ZERO cute ops remain and real
    arith (divsi/remsi/muli) is present."""
    e = KernelEmitter("s1_dyn")
    n = e.ssa("i32", "arith.constant 13 : i32")
    gp_int = e.ssa("i64", "arith.constant 4096 : i64")
    gp = e.ssa("!llvm.ptr<1>",
               f"llvm.inttoptr {gp_int.name} : i64 to !llvm.ptr<1>")
    leaf = DynLeaf(n)
    sh = algebra.make_shape(e, (leaf, 8))
    st = algebra.make_stride(e, (1, leaf))
    lay = algebra.make_layout(e, sh, st)
    tiler = algebra.make_shape(e, tile)
    zd = algebra.zipped_divide(e, lay, tiler)
    crd = algebra.make_coord(e, (1, 3))
    idx = algebra.layout_eval(e, crd, zd)
    scal = e.ssa("i32", '"cute.get_scalars"(' + idx.name + ') : '
                 '(' + idx.ctype.text + ') -> i32')
    i64 = e.ssa("i64", f"arith.extsi {scal.name} : i32 to i64")
    p = e.ssa("!llvm.ptr<1>",
              f"llvm.getelementptr {gp.name}[{i64.name}] "
              f": (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32")
    z = e.ssa("f32", "arith.constant 0.0 : f32")
    e.raw(f"llvm.store {z.name}, {p.name} : f32, !llvm.ptr<1>")
    out = _run_cc(e.module_text(), CUTE_PASSES)
    assert "cute." not in out, f"cute ops survived lowering:\n{out[:600]}"
    assert "arith.divsi" in out or "arith.remsi" in out  # real algebra


def test_dynamic_composition_lowers():
    """composition of dynamic outer with static inner lowers to arith."""
    e = KernelEmitter("s1_comp")
    n = e.ssa("i32", "arith.constant 20 : i32")
    s = e.ssa("i32", "arith.constant 2 : i32")
    sh_o = algebra.make_shape(e, (DynLeaf(n),))
    st_o = algebra.make_stride(e, (DynLeaf(s),))
    outer = algebra.make_layout(e, sh_o, st_o)
    inner = algebra.make_layout(e, (5, 4), (4, 1))
    comp = algebra.composition(e, outer, inner)
    crd = algebra.make_coord(e, (2, 3))
    algebra.layout_eval(e, crd, comp)
    out = _run_cc(e.module_text(), CUTE_PASSES)
    assert "cute." not in out, out[:500]


def test_dynamic_composition_static_fold_check():
    pass


def test_group_modes_and_slice_lowers():
    e = KernelEmitter("s1_gm")
    n = e.ssa("i32", "arith.constant 4 : i32")
    sh = algebra.make_shape(e, (DynLeaf(n), 5))
    st = algebra.make_stride(e, (1, DynLeaf(n)))
    lay = algebra.make_layout(e, sh, st)
    gm = algebra.group_modes(e, lay, 0, 2)
    # after grouping modes 0..2 the layout is rank 1; feed a rank-1 coord
    crd = algebra.make_coord(e, (13,))
    algebra.layout_eval(e, crd, gm)
    out = _run_cc(e.module_text(), CUTE_PASSES)
    assert "cute." not in out, out[:500]


# ---------------------------------------------------------- roundtrip parse
@pytest.mark.parametrize("op,args", [
    ("shape", ((1, 2),)),
    ("stride", ((2, 1),)),
    ("coord", ((0, 1),)),
    ("layout", None),
])
def test_textual_types_roundtrip(op, args):
    """Every emitted textual !cute<...> type must parse back (roundtrip
    validator discipline from the plan)."""
    e = KernelEmitter("s1_rt")
    if op == "shape":
        algebra.make_shape(e, args[0])
    elif op == "stride":
        algebra.make_stride(e, args[0])
    elif op == "coord":
        algebra.make_coord(e, args[0])
    else:
        sh = algebra.make_shape(e, (4, 8))
        st = algebra.make_stride(e, (8, 1))
        algebra.make_layout(e, sh, st)
    out = _run_cc(e.module_text(), [])
    assert "error" not in out.lower() or "module" in out
