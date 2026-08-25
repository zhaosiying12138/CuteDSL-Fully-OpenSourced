# Third-party provenance

## third_party/cutlass/ — BSD 3-Clause
- Source: git@github.com:NVIDIA/cutlass.git
- Commit: 7107b055 (full sha recorded in compat/sm120_toolchain.lock.yaml)
  ("[CuTeDSL] Fix export_to_c shape slot width for 64-bit dynamic dims (#3448)")
- Vendored subtrees (unmodified):
  - `cutlass_compiler/` — CuTe IR MLIR dialect stack (cute dialect, base facade,
    cute-opt/base-opt/cutlass-compiler tools, tests). Includes its own LICENSE.txt.
  - `examples/python/` — public CuTeDSL examples (BSD), incl. the
    `cute/blackwell_geforce` SM120 flagship demos used as conformance corpus.
- License: BSD-3-Clause — file `LICENSE` (copied from repo root LICENSE.txt).
- NOT vendored (deliberately, clean-room): `python/CuTeDSL` and any subtree
  governed by the NVIDIA Software License Agreement (EULA.txt). Nothing under
  EULA is read or copied by this project.

## third_party/flashinfer-src/ — Apache-2.0 (fetched, not committed)
- Source: git@github.com:flashinfer-ai/flashinfer.git
- Commit: 9d33a28e8321b2da099e7106fbc527ab3bca904c
- Fetched by `tools/fetch_third_party.sh` into `third_party/flashinfer-src/`
  (gitignored). Used as an *unmodified* system-test corpus: the SM120-compatible
  CuTeDSL operators (rmsnorm_fp4quant, add_rmsnorm_fp4quant, b12x MoE).
- License: Apache-2.0 — see LICENSE in that tree.

## Verification
`tools/verify_open_stack.py` checks that no EULA-governed artifact
(`python/CuTeDSL` sources, `_cutlass_ir` proprietary libs) is present in any
self-stack import path or build input.
