# dense_gemm.py (SM120 flagship) — compatibility gap analysis

Status: **NOT RUNNABLE yet on the self stack**. This document is the
faithful record of exactly what is missing (see anti-cheat rules: no
silent skips). Produced by static inventory of
`third_party/cutlass/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py`
(@ 7107b055, 1325 lines).

## Already implemented (self stack)

| Area | Evidence |
|---|---|
| `@cute.jit` / `@cute.kernel` / `cute.compile`, Constexpr specialization, dynamic scf.for/if | hello/elementwise/layout tests, 34 green |
| raw-pointer Tensor ABI, `from_dlpack`, elementwise tiled copy (make_tiled_copy_tv, partition_S, predicated copies, v4 vectorization) | official `elementwise_add.py` runs unmodified, 6 shapes golden |
| SMEM arrays, `ldmatrix.x4/x2.trans`, `mma.sync m16n8k16` with loop-carried accumulators | frontend warp-GEMM golden (3 shapes) |
| TMA G2S/S2G via TensorMap recipe (`cuTensorMapEncodeTiled` host-side), mbarrier init/expect_tx/try_wait.parity, multi-stage pipeline with phase rollover | raw + frontend TMA tests golden (incl. n_tiles=7 rollover) |
| `setmaxnreg inc/dec`, named barriers (`bar.sync/arrive`) | emission + execution verified on 5090 |

## Missing for dense_gemm.py unmodified (grouped, ~50 cute.* symbols)

1. **Layout algebra as first-class runtime values in kernels** —
   `cute.make_layout` inside @cute.kernel with dynamic modes,
   `ComposedLayout`, swizzle (`SW128` etc.), `make_layout_image_mask`,
   `cute.slice_`, `group_modes`, `zipped_divide` on tensors with
   composed layouts, `local_tile` with composed tilers. Our current
   layout layer is constexpr-meta only (host side).
2. **TMA atom plumbing as CuTe objects** —
   `cpasync.make_tiled_tma_atom` (builds a tiled TMA copy from a recipe
   + layout), `tma_partition` (source/destination partitioning),
   `CopyBulkTensorTileG2S(Multicast)Op/S2GOp` as atom objects,
   `prefetch_descriptor`. We have raw `cute.tma_load/tma_store` builtins
   instead of the atom/partition object model.
3. **TiledMma object model** — `make_tiled_mma(MmaF16BF16Op, atom_layout)`
   with `partition_A/B/C` returning fragment views wired to `cute.gemm`;
   our `gemm()` builtin is fixed-shape (16x8x16) without partitioning.
4. **TiledCopy variants** — `make_tiled_copy_A/B/C_atom/S` (MMA-atom
   derived copies), `StMatrix8x8x16bOp`/`LdMatrix8x8x8bOp` variants.
5. **`cutlass.pipeline` module** — `PipelineTmaAsync` class (producer
   acquire/release, full barrier arrays, cluster arrive/wait) as a
   Python driver object; our pipeline is raw mbarrier builtins.
6. **`cutlass.utils` helpers** — `blackwell_geforce` helpers,
   `hopper_helpers` (sm90_utils) used for epilogue/tile scheduler
   plumbing; `cute.struct.MemRange/Align` (typed SMEM allocator); the
   persistent scheduler loop; `cute.arch.cluster_*`, `fence_proxy`,
   `make_warp_uniform`, `warp_idx`, `block_idx_in_cluster`, `grid_dim`.
7. **Runtime side**: CUDA-graph-safe launch, `cuda.bindings.driver`
   stream plumbing used by the example's benchmark harness.

## Estimated effort

Items 1–4 are the deep part: they require the layout algebra to be
evaluated *inside the traced kernel* (dynamic layouts) rather than at
trace time. This is the core "CuTe DSL library" (~10k+ lines of Python
semantics in the official wheel). Item 5–6 are thinner wrappers over
primitives we already emit. Honest estimate: weeks of focused work
beyond the current session, per-milestone DoD as in the plan (M6 dense,
M7 blockscaled).

## What IS claimed today

- Official **elementwise_add.py runs unmodified** with golden checks on
  6 shapes including boundary predication (this is a full official
  kernel, not a toy).
- Every hardware primitive the flagship kernels use — TMA G2S/S2G,
  mbarrier with parity rollover multi-stage pipelines, ldmatrix,
  mma.sync m16n8k16, setmaxnreg, named barriers — is emitted by the
  self stack **and numerically verified** on the RTX 5090.
- Performance parity with the official compiler on the elementwise
  class (101–228% across shapes).

## Persistent multi-tile-per-CTA pipeline — RESOLVED (2026-08-26, 1ba06db)

The "residual" was TWO PTX memory-model violations, now fixed and
verified by 25x flake tests (0 fails) across multi-tile, refill, and
full-wave configurations up to 2048² (1024 CTAs):

1. Missing `fence.proxy.async.shared::cta` between generic SMEM stores
   and the TMA async-proxy read (S2G epilogue) — systematic corruption.
2. `mbarrier.init` executed by all threads on one object (PTX UB) —
   rare intermittent corruption at high CTA counts.

Persistent GEMM now: 9-config golden matrix + 31 TFLOP/s (72% of
official dense_gemm on this GPU). Remaining for dense_gemm.py
unmodified-run: the library-driver layer below.
