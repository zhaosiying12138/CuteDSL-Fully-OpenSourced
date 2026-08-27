"""Minimal `ir` surface for official-style code paths.

Official code uses cutlass._mlir.ir for type tokens (T.i32() etc.) and
dialectric access. The self stack's emitter works in textual MLIR, so a type
token is just its MLIR text; IrValue wraps an emitter SSA handle.
"""
from __future__ import annotations


class Type:
    """An MLIR type token (textual)."""

    def __init__(self, text: str):
        self.text = text

    def __str__(self):
        return self.text

    def __repr__(self):
        return f"Type({self.text})"

    def __eq__(self, other):
        return isinstance(other, Type) and self.text == other.text

    def __hash__(self):
        return hash(self.text)


class IrValue:
    """Wraps a self-stack SSA handle so ir_value() round-trips."""

    __slots__ = ("ssa", "const_val")

    def __init__(self, ssa, const_val=None):
        self.ssa = ssa
        self.const_val = const_val

    def __repr__(self):
        return f"IrValue({getattr(self.ssa, 'name', self.ssa)!r})"


class dialects:  # namespace: cutlass._mlir.ir.dialects.llvm
    pass
