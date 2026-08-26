"""cutlass.cute.nvgpu.warp — warp-level op markers (compat)."""


class MmaF16BF16Op:
    """mma.sync.aligned.m16n8k16 f16/bf16 atom.

    Official construction: MmaF16BF16Op(a_dtype, acc_dtype, mma_inst_mnk)
    — sm120a supports m16n8k16 (and k8 narrow variants); inst shapes are
    validated against the mma_atoms trait table at partition time.
    """

    def __init__(self, a_dtype=None, acc_dtype=None, mma_inst_mnk=None):
        self.a_dtype = a_dtype
        self.acc_dtype = acc_dtype
        self.mma_inst_mnk = mma_inst_mnk or (16, 8, 16)


class MmaMXF4Op:
    pass


class MmaMXF8Op:
    pass


class LdMatrix8x8x16bOp:
    """ldmatrix.x4.sync.aligned.m8n8.x16.shared.b16 warp copy op marker."""

    def __init__(self, trans=False, num=4):
        self.trans = bool(trans)
        self.num = num


class StMatrix8x8x16bOp:
    """stmatrix.x4 warp store op marker."""

    def __init__(self, trans=False, num=4):
        self.trans = bool(trans)
        self.num = num
