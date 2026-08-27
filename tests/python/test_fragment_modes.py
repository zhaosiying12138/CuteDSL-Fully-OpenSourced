"""Regression tests for shape-aware register-fragment mode slicing."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/cutlass_compat"))

import cutlass

from self_cutedsl.frontend import builtins
from self_cutedsl.frontend.cute_objects import AccumRetile
from self_cutedsl.frontend.emitter import KernelEmitter
from self_cutedsl.frontend.interp import KernelInterpreter
from self_cutedsl.frontend.meta import F32, cute_size


def test_shape_tuple_size_is_extent_product_not_rank():
    assert cute_size((128, 128)) == 16_384
    assert cute_size((128, 128, 128)) == 2_097_152


def test_rmem_tensor_preserves_shape_for_mode_queries():
    fragment = builtins.make_rmem_tensor((4, 2, 8), F32)

    assert fragment.shape == (4, 2, 8)
    assert cute_size(fragment) == 64
    assert cute_size(fragment, mode=[0]) == 4
    assert cute_size(fragment, mode=[1]) == 2
    assert cute_size(fragment, mode=[2]) == 8


def test_accum_retile_mode_slice_keeps_value_mode():
    """SM120 block-scaled retile speaks the official paired-8 fragment
    convention: two physical M atoms share one eight-value leading mode."""
    fragment = builtins.make_rmem_tensor((8, 1, 8), F32)
    fragment.slots.update({slot: f"slot-{slot}" for slot in range(64)})
    mma = SimpleNamespace(
        op=SimpleNamespace(shape_mnk=(16, 8, 64)),
        tile_mn=(64, 32),
    )
    retile = AccumRetile(SimpleNamespace(mma=mma), fragment)

    # default CTA tile (128,128) -> warp tile (32,64) -> paired (8, 1, 8)
    assert retile.shape == (8, 1, 8)
    assert cute_size(retile, mode=[1]) == 1
    assert cute_size(retile, mode=[2]) == 8

    atom = retile[(None, 0, 3)]
    assert atom.shape == (8,)
    assert cute_size(atom) == 8
    assert [atom[i] for i in range(8)] == [
        f"slot-{24 + i}" for i in range(8)
    ]

    atom[2] = "updated"
    assert fragment.slots[24 + 2] == "updated"


def test_typed_literal_arithmetic_stays_constexpr():
    interp = object.__new__(KernelInterpreter)
    interp.emitter = KernelEmitter("typed_literal")

    assert interp.binop(ast.Add(), cutlass.Int32(0), 1) == 1
    assert interp.binop(ast.Mod(), cutlass.Int32(5), 3) == 2


def test_byte_pack_uses_an_ssa_shift_operand():
    emitter = KernelEmitter("pack_bytes")
    lo = emitter.ssa("i8", "arith.constant 1 : i8")
    hi = emitter.ssa("i8", "arith.constant 2 : i8")

    packed = builtins._pack_bytes(emitter, lo, hi)
    text = "\n".join(emitter._lines)

    assert packed.type == "i32"
    assert "arith.constant 8 : i32" in text
    assert "llvm.shl" in text
    assert "llvm.shl %" in text
    assert "llvm.shl %v" in text and ", %v" in text
    assert "llvm.shl" not in next(
        line for line in text.splitlines() if "arith.constant 8" in line
    )


def test_blockscaled_mma_preserves_f32_accumulator_types():
    emitter = KernelEmitter("mxf4")
    packed = [emitter.ssa("i32", "arith.constant 0 : i32")
              for _ in range(10)]
    accum = [emitter.ssa("f32", "arith.constant 0.0 : f32")
             for _ in range(4)]

    outputs = emitter.mma_mxf4nvf4(
        packed[:4], packed[4:6], [packed[6]], [packed[8]], accum
    )
    mma_line = next(line for line in emitter._lines
                    if "kind::mxf4nvf4" in line)

    assert outputs != accum
    assert len(outputs) == 4 and all(value.type == "f32" for value in outputs)
    assert any("mxf4nvf4" in line for line in emitter._lines)
    assert "{$0, $1, $2, $3}" in mma_line
    assert "$rw" not in mma_line and "$r0" not in mma_line
    assert "ro (" in mma_line and "-> !llvm.struct<(f32, f32, f32, f32)>" in mma_line
    assert mma_line.count("i32") == 12 and mma_line.count("i16") == 4
    assert "-> !llvm.struct<(f32, f32, f32, f32)>" in mma_line
