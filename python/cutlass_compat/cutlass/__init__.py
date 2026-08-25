"""cutlass — CuTeDSL-Fully-OpenSourced compatibility namespace.

API-compatible surface for the proprietary `nvidia-cutlass-dsl` `cutlass`
package, implemented entirely by the open self stack. Programs written for
the official DSL import `cutlass` and `cutlass.cute`; with this package on
sys.path (and the official wheel NOT installed) they run on our compiler.

The compat marker lets conformance tooling distinguish us:
    cutlass.__self_cutedsl__ is True
"""
from . import cute  # noqa: F401  (cutlass.cute)
from . import torch  # noqa: F401  (cutlass.torch)
from .dtypes import (  # noqa: F401
    Boolean,
    Constexpr,
    Int32,
    Int64,
    UInt32,
    Float32,
    Float16,
    BFloat16,
    Float8E4M3FN,
    Float8E5M2,
    Float8E8M0FNU,
    Float4E2M1FN,
    Numeric,
    dtype,
)
from .cuda import initialize_cuda_context  # noqa: F401

__self_cutedsl__ = True

__version__ = "0.1.0+compat"
