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


class MmaMXF4NVF4Op:
    """mma.sync m16n8k64 row.col kind::mxf4nvf4 (fp4 x fp4, e4m3 SF,
    sf_vec 16) — nvfp4 class."""

    shape_mnk = (16, 8, 64)

    def __init__(self, a_dtype=None, acc_dtype=None, sf_dtype=None):
        self.a_dtype = a_dtype
        self.acc_dtype = acc_dtype
        self.sf_dtype = sf_dtype
        self.sf_vec_size = 16


class MmaMXF4Op:
    """mma.sync m16n8k64 kind::mxf4 (fp4 x fp4, e8m0 SF, sf_vec 32)."""

    shape_mnk = (16, 8, 64)

    def __init__(self, a_dtype=None, acc_dtype=None, sf_dtype=None):
        self.a_dtype = a_dtype
        self.acc_dtype = acc_dtype
        self.sf_dtype = sf_dtype
        self.sf_vec_size = 32


class MmaMXF8F6F4Op:
    """mma.sync m16n8k32 kind::mxf8f6f4 (fp8/fp6/fp4 mixed, e8m0 SF,
    sf_vec 32)."""

    shape_mnk = (16, 8, 32)

    def __init__(self, a_dtype=None, acc_dtype=None, sf_dtype=None):
        self.a_dtype = a_dtype
        self.acc_dtype = acc_dtype
        self.sf_dtype = sf_dtype
        self.sf_vec_size = 32


class LdMatrix8x16x8bOp:
    """ldmatrix.b4x16_p64-class 8-bit unpack load (FP4-in-Int8 path)."""

    def __init__(self, transpose=False, num_matrices=4, unpack_bits=None):
        self.trans = bool(transpose)
        self.num = num_matrices
        self.unpack_bits = unpack_bits
