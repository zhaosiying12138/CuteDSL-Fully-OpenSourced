# tools/perf — dual-stack performance suite

One methodology for every operator family: the **same unmodified operator
source** runs twice — once on the self stack (`.venv-self` + this repo's
`python/` on `PYTHONPATH`) and once on the official wheel
(`.venv-reference`) — then `render_tables.py` merges the captures.

| Script | Family | Operator source (unmodified) |
|---|---|---|
| `bench_elementwise.py` | elementwise add FP32 | CUTLASS ampere demo `elementwise_add.py` |
| `bench_dense_gemm.py` | dense FP16 GEMM | CUTLASS blackwell_geforce `dense_gemm.py` |
| `bench_blockscaled.py` | NVFP4 blockscaled GEMM | CUTLASS blackwell_geforce cooperative demo |
| `bench_flashinfer_norm.py` | rmsnorm/add-rmsnorm + NVFP4 quant | flashinfer `cute_dsl/{rmsnorm,add_rmsnorm}_fp4quant.py` |
| `bench_flashinfer_b12x.py` | W4A16 fused MoE | flashinfer `fused_moe/cute_dsl/b12x_moe.py` + `blackwell_sm12x/*` |

**No correctness verification happens here** — that is the pytest suite
(`tools/run_correctness.sh`). These scripts are pure performance + baseline
comparison.

## Timing methodology

* CUTLASS demos: their own `testing.benchmark` harness (CUDA events,
  µs/iteration; `--warmup 10 --iters 50` defaults here).
* flashinfer operators: per-iteration CUDA event pairs driven by
  `perf_common.cuda_time_us` (median / p10 / p90; `--warmup 10 --iters 50`,
  b12x `3/20`).
* Every capture stores the GPU state (utilization, clocks, power, memory)
  next to the numbers, so shared-GPU captures are honestly labeled.

## Running

```bash
tools/perf/run_all_perf.sh                    # everything, both stacks
.venv-self/bin/python tools/perf/bench_dense_gemm.py   # one family, one stack
.venv-self/bin/python tools/perf/render_tables.py      # re-merge tables
```

Outputs land in `artifacts/perf/`: `<family>_{self,official}.json` (raw,
per-stack), `summary.md` / `summary.json` (merged tables + headline mean),
and refreshed legacy combined files for `dense_gemm` / `blockscaled`.

## Formal (exclusive-GPU) re-capture

1. Confirm the GPU is otherwise idle: `nvidia-smi dmon -c 3` (on WSL:
   `/usr/lib/wsl/lib/nvidia-smi dmon -c 3`) should show ~0% SM.
2. `tools/perf/run_all_perf.sh`
3. `tools/perf/render_tables.py`
4. Update the tables in `README.md` / `README_EN.md` and the blog from
   `artifacts/perf/summary.md`.

## Headline convention

per-shape pct = `official_us / self_us × 100`; family pct = mean over its
shapes; headline = arithmetic mean over the six official-baselined families.
MLA (no official sm_120a implementation exists) and the self-built
persistent-GEMM milestones are reported separately and never enter the
headline.
