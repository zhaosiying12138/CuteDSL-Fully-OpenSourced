#!/usr/bin/env python3
"""inspect_ptx.py — audit PTX produced by the self stack.

The self stack loads textual PTX straight into the CUDA driver JIT
(``DG_DUMP_PTX=1`` drops every compiled module to the platform temporary
directory as ``dg_mod_<n>.ptx``).
This tool walks those dumps (or any ``--glob``) and verifies, per module:

  * the PTX target is exactly ``sm_120a`` (the only supported profile);
  * the module exposes at least one ``.entry`` kernel;
  * no ptxas/nvcc artifacts appear (the self path never runs them — PTX
    comes from the in-tree LLVM NVPTX backend via cutlass-compiler);
  * tensor-core kernels really contain MMA instructions (anti-masquerade
    check: no scalar-FMA GEMMs posing as ``mma.sync``).

Exit code is non-zero if any check fails, so it can gate CI/regression.

Usage:
  # run any self-stack workload with dumps enabled, then:
  DG_DUMP_PTX=1 .venv-self/bin/python -m pytest -m sm120 -k dense_gemm
  .venv-self/bin/python tools/inspect_ptx.py                 # scan the temp dir
  .venv-self/bin/python tools/inspect_ptx.py --glob 'artifacts/ptx/*.ptx' --strict-mma
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
import tempfile
from pathlib import Path

TARGET_LINE = re.compile(r"^\.target\s+(\S+)", re.MULTILINE)
ENTRY_RE = re.compile(r"^\.visible\s+\.entry\s+(\S+)", re.MULTILINE)
MMA_RE = re.compile(r"\bmma\.sync", re.MULTILINE)
# ptxas-processed PTX carries "//" compile stamps from the CUDA toolkit;
# raw MLIR-NVPTX output does not.
TOOLKIT_STAMP_RE = re.compile(r"^\s*//\s*(ptxas|nvcc|Compiled by)", re.MULTILINE)


def inspect_module(path: str) -> dict:
    text = Path(path).read_text(errors="replace")
    targets = TARGET_LINE.findall(text)
    entries = ENTRY_RE.findall(text)
    return {
        "path": path,
        "target": targets[0] if targets else None,
        "entries": entries,
        "n_lines": text.count("\n"),
        "has_mma": bool(MMA_RE.search(text)),
        "toolkit_stamp": bool(TOOLKIT_STAMP_RE.search(text)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_glob = str(Path(tempfile.gettempdir()) / "dg_mod_*.ptx")
    ap.add_argument("--glob", default=default_glob,
                    help=f"glob of PTX dumps (default: {default_glob})")
    ap.add_argument("--strict-mma", action="store_true",
                    help="require every module to contain mma.sync (MMA-heavy "
                         "workloads only)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"no PTX found for {args.glob!r} — run your workload with "
              f"DG_DUMP_PTX=1 first")
        return 2

    failures = []
    print(f"{'module':<28} {'target':<10} {'entries':>7} {'lines':>8}  mma  stamp")
    for p in paths:
        info = inspect_module(p)
        name = Path(p).name
        mma = "yes" if info["has_mma"] else "-"
        stamp = "TOOLKIT" if info["toolkit_stamp"] else "-"
        print(f"{name:<28} {str(info['target']):<10} {len(info['entries']):>7} "
              f"{info['n_lines']:>8}  {mma:<4} {stamp}")
        if info["target"] != "sm_120a":
            failures.append(f"{name}: target {info['target']!r} != sm_120a")
        if not info["entries"]:
            failures.append(f"{name}: no .entry kernels")
        if info["toolkit_stamp"]:
            failures.append(f"{name}: toolkit compile stamp present "
                            f"(ptxas/nvcc must not touch the self path)")
        if args.strict_mma and not info["has_mma"]:
            failures.append(f"{name}: no mma.sync under --strict-mma")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"\nOK: {len(paths)} module(s), all .target sm_120a, "
          f"no toolkit stamps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
