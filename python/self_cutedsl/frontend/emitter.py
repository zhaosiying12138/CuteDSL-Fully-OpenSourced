"""emitter.py — MLIR text emitter for traced kernels (M2 scope).

Emits gpu.func bodies in the base-dialect boundary (arith/scf/gpu/LLVM),
which cutlass-compiler's one-shot-convert-to-llvm lowers to NVVM/PTX.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(eq=False)
class SSA:
    """A dynamic (runtime) value: an SSA result with an MLIR type."""
    type: str
    id: int
    dtype_name: str = "int32"

    @property
    def name(self) -> str:
        return f"%v{self.id}"

    def __repr__(self):
        return f"<SSA {self.name}:{self.type}>"


class KernelEmitter:
    """Accumulates the body of one gpu.func."""

    def __init__(self, name: str, indent: str = "    "):
        self.name = name
        self._indent = indent
        self._lines: list[str] = []
        self._depth = 1  # inside gpu.func
        self._next_id = 0
        self.params: list[tuple[str, str]] = []  # (mlir_name, type)

    # -- plumbing ----------------------------------------------------------
    def raw(self, line: str) -> None:
        self._lines.append(self._indent * self._depth + line)

    def ssa(self, type_: str, op: str, dtype_name: str = "int32") -> SSA:
        v = SSA(type_, self._next_id, dtype_name)
        self._next_id += 1
        self.raw(f"{v.name} = {op}")
        return v

    def const_i32(self, value: int) -> SSA:
        return self.ssa("i32", f"arith.constant {value} : i32")

    def const_i64(self, value: int) -> SSA:
        return self.ssa("i64", f"arith.constant {value} : i64")

    # -- structured control flow --------------------------------------------
    def open_if(self, cond: SSA) -> None:
        assert cond.type == "i1"
        self.raw(f"scf.if {cond.name} {{")
        self._depth += 1

    def close_if(self) -> None:
        self._depth -= 1
        self.raw("}")

    def open_for(self, lb: SSA, ub: SSA, step: SSA) -> SSA:
        iv = SSA("index", self._next_id, "index")
        self._next_id += 1
        self.raw(f"scf.for {iv.name} = {lb.name} to {ub.name} step {step.name} {{")
        self._depth += 1
        return iv

    def close_for(self) -> None:
        self._depth -= 1
        self.raw("}")

    # -- memory (raw pointer ABI; tensors lower to !llvm.ptr<1>) --------------
    def index_to_i64(self, v: SSA) -> SSA:
        if v.type == "i64":
            return v
        assert v.type == "index", f"index_to_i64 on {v.type}"
        return self.ssa("i64", f"arith.index_cast {v.name} : index to i64")

    def gep(self, base: SSA, offset: SSA, elem_type: str = "f32") -> SSA:
        off = self.index_to_i64(offset) if offset.type in ("index", "i32") else offset
        return self.ssa(
            base.type,
            f"llvm.getelementptr {base.name}[{off.name}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, {elem_type}",
        )

    def load_f32(self, p: SSA) -> SSA:
        return self.ssa("f32", f"llvm.load {p.name} : !llvm.ptr<1> -> f32", "float32")

    def store_f32(self, v: SSA, p: SSA) -> None:
        self.raw(f"llvm.store {v.name}, {p.name} : f32, !llvm.ptr<1>")

    # -- builtins -------------------------------------------------------------
    def thread_id(self, axis: str) -> SSA:
        return self.ssa("index", f"gpu.thread_id {axis}", "index")

    def block_id(self, axis: str) -> SSA:
        return self.ssa("index", f"gpu.block_id {axis}", "index")

    def block_dim(self, axis: str) -> SSA:
        return self.ssa("index", f"gpu.block_dim {axis}", "index")

    def printf(self, fmt_c: str, args: list[SSA]) -> None:
        escaped = fmt_c.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\0A")
        if args:
            ops = ", ".join(a.name for a in args)
            types = ", ".join(a.type for a in args)
            self.raw(f'gpu.printf "{escaped}", {ops} : {types}')
        else:
            self.raw(f'gpu.printf "{escaped}"')

    # -- output ---------------------------------------------------------------
    def module_text(self) -> str:
        # param SSAs are pre-allocated as %v{i} by the interpreter; keep names in sync
        params = ", ".join(f"%v{i} : {ty}" for i, (_, ty) in enumerate(self.params))
        lines = [
            "module attributes {gpu.container_module} {",
            f"  gpu.module @{self.name}_module {{",
            f"    gpu.func @{self.name}({params}) kernel {{",
            *self._lines,
            "      gpu.return",
            "    }",
            "  }",
            "}",
        ]
        return "\n".join(lines) + "\n"
