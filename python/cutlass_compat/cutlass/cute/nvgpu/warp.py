"""cutlass.cute.nvgpu.warp — warp-level op markers (compat)."""


class MmaF16BF16Op:
    """mma.sync.aligned.m16n8k16 f16/bf16 atom."""

    def __init__(self, acc_dtype=None):
        self.acc_dtype = acc_dtype


class MmaMXF4Op:
    pass


class MmaMXF8Op:
    pass
