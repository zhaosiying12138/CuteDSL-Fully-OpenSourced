"""cutlass.cute.nvgpu — device op markers (compat)."""


class CopyUniversalOp:
    """Plain universal load/store (ld.global/st.global)."""


class LdMatrixOp:
    pass


class StMatrixOp:
    pass
