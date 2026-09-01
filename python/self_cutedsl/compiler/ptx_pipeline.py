"""ptx_pipeline.py — drive the open MLIR → sm_120a PTX pipeline from Python.

Stages (all open-source):
  1. cutlass-compiler (BSD) : one-shot-convert-to-llvm
                             attach-nvvm-target=chip=sm_120a
                             emit-gpu-binary=compilation-target=isa
  2. extract the textual PTX from the resulting gpu.binary attribute.

No nvcc / NVRTC / ptxas / libNVVM involved: `emit-gpu-binary` in `isa` mode
serializes via the in-tree LLVM NVPTX backend.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CUTLASS_COMPILER = ROOT / "build-compiler/tools/cutlass-compiler/cutlass-compiler"

DEFAULT_PASSES = [
    "--one-shot-convert-to-llvm",
    "--attach-nvvm-target=chip=sm_120a",
    "--emit-gpu-binary=compilation-target=isa",
]

_ASSEMBLY_RE = re.compile(r'assembly = "((?:[^"\\]|\\.)*)"')


def _compiler_command() -> Path:
    """Return the native cutlass-compiler executable path."""
    candidate = CUTLASS_COMPILER
    if os.name == "nt" and not candidate.exists():
        candidate = candidate.with_suffix(".exe")
    return candidate


def _debug_file(name: str) -> Path:
    return Path(tempfile.gettempdir()) / name


def _unescape(s: str) -> str:
    return (
        s.replace("\\0A", "\n")
        .replace("\\09", "\t")
        .replace("\\22", '"')
        .replace("\\5C", "\\")
        .replace('\\"', '"')
    )


def compile_mlir_to_ptx(mlir_text: str | Path, extra_passes: list[str] | None = None) -> str:
    """Compile an MLIR module (gpu.container_module) to textual sm_120a PTX."""
    if isinstance(mlir_text, Path):
        mlir_text = mlir_text.read_text()
    cmd = [str(_compiler_command()), *DEFAULT_PASSES, *(extra_passes or [])]
    proc = subprocess.run(cmd, input=mlir_text, capture_output=True, text=True)
    if proc.returncode != 0:
        import os as _os
        if _os.environ.get("DG_DUMP_MLIR"):
            _debug_file("fail_mod.mlir").write_text(mlir_text)
        raise RuntimeError(
            f"cutlass-compiler failed ({proc.returncode}):\n{proc.stderr[:4000]}")
    import os as _os2
    if _os2.environ.get("DG_DUMP_MLIR_OK"):
        _debug_file("ok_mod.mlir").write_text(mlir_text)
    out = proc.stdout

    targets = re.findall(r'#nvvm\.target<chip = "([^"]+)"', out)
    if targets and any(t != "sm_120a" for t in targets):
        raise RuntimeError(f"non-sm_120a target emitted: {targets}")

    m = _ASSEMBLY_RE.search(out)
    if not m:
        raise RuntimeError(f"no gpu.binary assembly in output:\n{out[:2000]}")
    ptx = _unescape(m.group(1))

    if ".target sm_120a" not in ptx:
        raise RuntimeError("emitted PTX is not .target sm_120a")
    if ".entry" not in ptx:
        raise RuntimeError("emitted PTX contains no kernel entry")
    return ptx


def entry_names(ptx: str) -> list[str]:
    return re.findall(r"\.visible \.entry ([A-Za-z_.$][\w.$]*)", ptx)
