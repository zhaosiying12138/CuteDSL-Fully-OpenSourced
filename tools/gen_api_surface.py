#!/usr/bin/env python3
"""gen_api_surface.py — inventory the CuTeDSL API surface used by the frozen corpus.

Statically walks the conformance corpus Python files (CUTLASS blackwell_geforce
demos + FlashInfer SM120 operators) and extracts every attribute access rooted
at `cutlass` / `cute` / module aliases imported from them. Emits
compat/sm120_api_surface.yaml: symbol -> files using it -> usage count.

This defines the API surface the self frontend must eventually cover.

Usage: .venv-self/bin/python tools/gen_api_surface.py
"""
from __future__ import annotations

import ast
import collections
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

CORPUS = {
    "cutlass_demos": [
        "third_party/cutlass/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py",
        "third_party/cutlass/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_cooperative.py",
        "third_party/cutlass/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_pingpong.py",
        "third_party/cutlass/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/blockscaled_gemm_dispatch.py",
    ],
    "flashinfer_ops": [
        "third_party/flashinfer-src/flashinfer/cute_dsl/rmsnorm_fp4quant.py",
        "third_party/flashinfer-src/flashinfer/cute_dsl/add_rmsnorm_fp4quant.py",
        "third_party/flashinfer-src/flashinfer/cute_dsl/fp4_common.py",
        "third_party/flashinfer-src/flashinfer/cute_dsl/utils.py",
        "third_party/flashinfer-src/flashinfer/fused_moe/cute_dsl/b12x_moe.py",
        "third_party/flashinfer-src/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_micro_kernel.py",
        "third_party/flashinfer-src/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_static_kernel.py",
        "third_party/flashinfer-src/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py",
        "third_party/flashinfer-src/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py",
        "third_party/flashinfer-src/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_direct_micro_kernel.py",
        "third_party/flashinfer-src/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_w4a16_kernel.py",
    ],
}


def module_alias_map(tree: ast.AST) -> dict:
    """Map local alias -> dotted module path for cutlass/cute-rooted imports."""
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root in ("cutlass",):
                    aliases[a.asname or a.name] = a.name
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == "cutlass" or node.module.startswith("cutlass.")):
                for a in node.names:
                    aliases[a.asname or a.name] = f"{node.module}.{a.name}"
    return aliases


def dotted_chain(node: ast.AST) -> list[str] | None:
    """Flatten an Attribute/Name chain into its parts, outermost last."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        parts.reverse()
        return parts
    return None


def collect(paths) -> dict:
    symbols = collections.defaultdict(lambda: {"count": 0, "files": collections.Counter()})
    for p in paths:
        path = ROOT / p
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(errors="replace"))
        alias = module_alias_map(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            chain = dotted_chain(node)
            if not chain:
                continue
            # Resolve any alias-rooted chain back to its full dotted path.
            base = alias.get(chain[0])
            if base:
                full = ".".join([base] + chain[1:])
            elif chain[0] in ("cutlass",):
                full = ".".join(chain)
            else:
                continue
            symbols[full]["count"] += 1
            symbols[full]["files"][p] += 1
    return symbols


def main() -> None:
    surface = {}
    for group, relpaths in CORPUS.items():
        syms = collect(relpaths)
        surface[group] = {
            sym: {"count": info["count"], "files": dict(info["files"])}
            for sym, info in sorted(syms.items())
        }
        print(f"[{group}] {len(syms)} distinct symbols")

    # Global rollup: symbols needed by BOTH groups are the compatibility core.
    all_syms = collections.defaultdict(set)
    for group, syms in surface.items():
        for sym in syms:
            all_syms[sym].add(group)
    core = sorted(s for s, g in all_syms.items() if len(g) == len(CORPUS))

    out = {
        "schema": 1,
        "generated_by": "tools/gen_api_surface.py",
        "note": "API surface required by the frozen SM120 conformance corpus",
        "shared_core_symbols": core,
        **surface,
    }
    dest = ROOT / "compat/sm120_api_surface.yaml"
    dest.write_text(yaml.safe_dump(out, sort_keys=False, width=100))
    print(f"wrote {dest}: {sum(len(v) for v in surface.values())} symbols, "
          f"{len(core)} shared by all groups")


if __name__ == "__main__":
    main()
