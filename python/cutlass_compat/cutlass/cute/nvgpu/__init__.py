"""cutlass.cute.nvgpu — device op markers (compat)."""


class CopyUniversalOp:
    """Plain universal load/store (ld.global/st.global)."""


class LdMatrixOp:
    pass


class StMatrixOp:
    pass


def __getattr__(name):
    if name == "warp":
        from importlib import import_module

        return import_module("cutlass.cute.nvgpu.warp")
    raise AttributeError(name)
