"""cutlass._mlir bridge: official CuTeDSL programs reach MLIR through this
namespace (ir values, llvm dialect ops). The self stack emits TEXTUAL MLIR,
so `ir_value()` here wraps the emitter's SSA handles; llvm.inline_asm lowers
to nvvm.inline_ptx with explicit result types (the encoding proven by the
blockscaled mxf4nvf4 path).
"""
from . import ir  # noqa: F401
from .dialects import llvm  # noqa: F401
