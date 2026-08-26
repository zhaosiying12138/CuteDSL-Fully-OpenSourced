"""cutlass.cute.nvgpu — device op markers (compat)."""


class CopyUniversalOp:
    """Plain universal load/store (ld.global/st.global)."""


class LdMatrixOp:
    pass


class StMatrixOp:
    pass


def __getattr__(name):
    from importlib import import_module

    if name in ("warp", "cpasync", "warpgroup"):
        return import_module(f"cutlass.cute.nvgpu.{name}")
    raise AttributeError(name)
