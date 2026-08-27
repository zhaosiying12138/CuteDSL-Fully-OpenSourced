# SBOM — CuteDSL-Fully-OpenSourced (sm120 profile)

## First-party (this repo, BSD-3-Clause)
| Component | Path | Notes |
|---|---|---|
| self_cutedsl frontend (AST trace → textual MLIR) | python/self_cutedsl/frontend | no `_cutlass_ir` dependency |
| CuTe object model (algebra via C++ passes) | python/self_cutedsl/object_model | mma_atoms trait table, pipeline drivers |
| runtime (Driver JIT, tensor-map encode, manifests) | python/self_cutedsl/runtime | cuModuleLoadDataEx + cuTensorMapEncodeTiled only |
| cutlass_compat (BSD-licensed surface re-export layer) | python/cutlass_compat | maps official wheel API onto self stack |
| tests | tests/ | 121 tests incl. verbatim flagship kernels (115 host/python + 6 on-GPU runtime) |

## Third-party
| Component | Version/Commit | License | Use |
|---|---|---|---|
| cutlass (examples only) | vendored @ 7107b055-equivalent subtree | BSD-3 | kernel sources under test (dense_gemm, blockscaled, elementwise) |
| cutlass_compiler (cute dialect + passes + tools) | vendored subtree, LLVM pin 23a60f15 | BSD-3 | cute.* folding/expand/to-base + LLVM NVPTX backend |
| cutlass C++ headers (cute arch/atom) | sparse clone of NVIDIA/cutlass@main | BSD-3 | authoritative SM120 mma asm/trait reference |
| torch (self env) | 2.x cu13 | BSD-3 | tensors, reference golden, timing |
| cuda-python / CUDA toolkit | 13.3 | CUDA EULA (driver API only) | driver JIT, sanitizer |
| flashinfer | not vendored (baseline only) | Apache-2.0 | reference-env acceptance pool |

## Anti-cheat attestations
- No `_cutlass_ir` / nvcc / NVRTC / ptxas invocation in the self stack
  (tools/verify_open_stack.py asserts the import/PATH surface;
  tools/inspect_ptx.py audits PTX dumps — target/entries/MMA presence —
  for any workload run with DG_DUMP_PTX=1; ptxas was used only as a
  diagnostic in /tmp during debugging, never in the build path).
- No filename/shape/argv special-casing; no scalar-FMA masquerade (audited
  PTX contains nvvm.mma.sync / OMMA.SF only); no hand-encoded
  CUtensorMap in kernels (all via cuTensorMapEncodeTiled); tolerances
  never loosened beyond the official runner's own.
- Official wheel exists only in the isolated reference env; its PTX output
  was read as a comparison oracle (public ISA), never its implementation.
