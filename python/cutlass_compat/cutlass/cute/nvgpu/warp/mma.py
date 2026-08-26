"""cutlass.cute.nvgpu.warp.mma — SM120 block-scaled MMA op catalog."""

# operand-dtype pairs accepted by mma.sync.kind::mxf8f6f4 on sm_120a
# (a_dtype, b_dtype) tuples; the dispatch module consults this table.
import cutlass

MXF8F6F4_SUPPORTED_PAIRS = (
    (cutlass.Float8E4M3FN, cutlass.Float8E4M3FN),
    (cutlass.Float8E5M2, cutlass.Float8E5M2),
    (cutlass.Float4E2M1FN, cutlass.Float8E4M3FN),
    (cutlass.Float8E4M3FN, cutlass.Float4E2M1FN),
)
