"""cuda_driver.py — CUDA Driver JIT runtime for sm_120a textual PTX.

Loads PTX produced by the open compiler stack (cutlass-compiler passes +
LLVM NVPTX) via cuModuleLoadDataEx and launches kernels via cuLaunchKernel.
No nvcc / NVRTC / ptxas / libNVVM anywhere in this path: the driver's JIT
compiles the textual PTX on-device.

Memory allocation intentionally stays with torch tensors (DLPack-compatible
host buffers whose device pointers we pass to the kernel); this mirrors how
CuTeDSL host code manages tensors.
"""
from __future__ import annotations

import ctypes
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

try:  # cuda-python < 13 layout
    from cuda import cuda as cu
except ImportError:  # cuda-bindings >= 13 layout
    from cuda.bindings import driver as cu

TARGET = "sm_120a"
PTX_VERSION = "8.7"


def _check(err: "cu.CUresult | tuple", api: str) -> None:
    if isinstance(err, tuple):  # cuda-bindings >= 13 returns (CUresult, ...)
        err = err[0]
    if int(err) != int(cu.CUresult.CUDA_SUCCESS):
        raise RuntimeError(f"{api} failed: {err}")


@dataclass
class LaunchManifest:
    """Per-PTX launch plan emitted by the compiler side.

    Schema (see compat docs): entry, args, block/grid formulas or explicit
    dims, dynamic smem, cluster dims, tensor-map args, target, ptx_version.
    """

    entry: str
    args: list[dict] = field(default_factory=list)   # {name, type: "ptr"/"i32"/...}
    block: tuple = (1, 1, 1)
    grid: tuple = (1, 1, 1)
    dynamic_smem_bytes: int = 0
    cluster: tuple = (1, 1, 1)
    tensor_maps: list = field(default_factory=list)
    target: str = TARGET
    ptx_version: str = PTX_VERSION

    @staticmethod
    def from_json(path: str | Path) -> "LaunchManifest":
        d = json.loads(Path(path).read_text())
        return LaunchManifest(**d)


class DriverJit:
    """Owns CUDA init + one module loaded from textual PTX."""

    def __init__(self, ptx: str | bytes):
        self._func_cache: dict = {}
        if isinstance(ptx, str):
            ptx = ptx.encode() + b"\x00"  # NUL-terminated image (driver reads to NUL)
        # Sanity: our one and only target.
        head = ptx[:512].decode(errors="replace")
        if ".target sm_120a" not in head:
            raise ValueError(f"refusing to load PTX that is not .target sm_120a:\n{head}")
        _check(cu.cuInit(0), "cuInit")
        err, dev = cu.cuDeviceGet(0)
        _check(err, "cuDeviceGet")
        err, self._ctx = cu.cuDevicePrimaryCtxRetain(dev)
        _check(err, "cuDevicePrimaryCtxRetain")
        _check(cu.cuCtxSetCurrent(self._ctx), "cuCtxSetCurrent")
        import os as _os
        if _os.environ.get("DG_USE_PTX"):
            with open(_os.environ["DG_USE_PTX"], "rb") as _f:
                ptx = _f.read() + b"\0"
        elif _os.environ.get("DG_DUMP_PTX"):
            _n = getattr(DriverJit, "_dump_n", 0) + 1
            DriverJit._dump_n = _n
            with open(f"/tmp/dg_mod_{_n}.ptx", "w") as f:
                f.write(ptx if isinstance(ptx, str) else ptx.decode())
        err, self._mod = cu.cuModuleLoadDataEx(ptx, 0, [], [])
        _check(err, "cuModuleLoadDataEx")

    def launch(self, manifest: LaunchManifest, *args: Any,
               stream=None, grid=None, block=None) -> None:
        func = self._func_cache.get(manifest.entry)
        if func is None:
            err, func = cu.cuModuleGetFunction(self._mod, manifest.entry.encode())
            _check(err, "cuModuleGetFunction")
            self._func_cache[manifest.entry] = func
        grid = grid or manifest.grid
        block = block or manifest.block
        packed = _pack_args(manifest, args)
        stream_obj = stream if stream is not None else cu.CUstream(0)
        _check(
            cu.cuLaunchKernel(
                func,
                *grid, *block,
                manifest.dynamic_smem_bytes,
                stream_obj,
                packed,
                0,
            ),
            "cuLaunchKernel",
        )

    def synchronize(self) -> None:
        _check(cu.cuCtxSynchronize(), "cuCtxSynchronize")

    @staticmethod
    def synchronize_ctx() -> None:
        _check(cu.cuCtxSynchronize(), "cuCtxSynchronize")


def _pack_args(manifest: LaunchManifest, args: Sequence[Any]) -> ctypes.Array:
    """Pack kernel args per manifest signature into a cuLaunchKernel arg buffer."""
    holders = []
    for spec, val in zip(manifest.args, args):
        t = spec["type"]
        if t == "ptr":
            ptr = _device_ptr(val)
            holders.append((ctypes.c_void_p(ptr), ctypes.c_void_p))
        elif t in ("i32", "u32"):
            holders.append((ctypes.c_uint32(int(val)), ctypes.c_uint32))
        elif t in ("i64", "u64"):
            holders.append((ctypes.c_uint64(int(val)), ctypes.c_uint64))
        elif t == "f32":
            holders.append((ctypes.c_float(float(val)), ctypes.c_float))
        elif t == "f64":
            holders.append((ctypes.c_double(float(val)), ctypes.c_double))
        else:
            raise ValueError(f"unsupported arg type {t!r}")
    arr = (ctypes.c_void_p * len(holders))()
    for i, (obj, _ty) in enumerate(holders):
        arr[i] = ctypes.cast(ctypes.pointer(obj), ctypes.c_void_p)
    # keep the python objects alive alongside the array
    arr._holders = holders  # type: ignore[attr-defined]
    return arr


def _device_ptr(val: Any) -> int:
    """torch tensor (device memory) -> raw pointer."""
    ptr = getattr(val, "data_ptr", None)
    if ptr is not None:
        return int(ptr())
    raise TypeError(f"cannot derive device pointer from {type(val)}")
