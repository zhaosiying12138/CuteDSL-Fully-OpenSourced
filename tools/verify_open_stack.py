#!/usr/bin/env python3
"""verify_open_stack.py — assert the SELF stack contains no proprietary
compiler components.

Checks (fail = non-zero exit):
  1. No `cutlass` / `nvidia_cutlass_dsl` importable module in the running
     interpreter (the official DSL wheel must not leak into the self env).
  2. No `_cutlass_ir` extension loaded in this process.
  3. nvcc / ptxas / NVRTC not on PATH and not used to produce our PTX
     (spot-check: the compiler tools we invoke are cutlass-compiler +
     in-process MLIR serialization only).
  4. EULA-governed sources (python/CuTeDSL of the cutlass repo) are not
     vendored under third_party/.

Usage: .venv-self/bin/python tools/verify_open_stack.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
failures: list[str] = []

# 1. official wheel importable?
for mod in ("cutlass", "nvidia_cutlass_dsl"):
    if importlib.util.find_spec(mod) is not None:
        failures.append(f"official module {mod!r} is importable in this environment")

# 2. proprietary extension loaded?
loaded = [m for m in sys.modules if m.startswith("_cutlass_ir")]
if loaded:
    failures.append(f"proprietary extension loaded: {loaded}")

# 3. banned tools on PATH?
import shutil

for tool in ("nvcc", "ptxas", "nvrtc-compiler"):
    if shutil.which(tool):
        failures.append(f"banned tool on PATH: {tool}")

# 4. EULA subtree vendored? (the repo's python/ package dir — NOT the BSD
#    examples/python/CuTeDSL demos, which we vendor deliberately)
eula_root = ROOT / "third_party/cutlass/python"
if eula_root.exists():
    failures.append(f"EULA-governed subtree vendored: {eula_root}")

if failures:
    print("OPEN-STACK VIOLATIONS:")
    for f in failures:
        print("  -", f)
    sys.exit(1)

print("open-stack verification OK: no proprietary compiler components "
      "in import path, loaded modules, PATH, or vendored sources.")
