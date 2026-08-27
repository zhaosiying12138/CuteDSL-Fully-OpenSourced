#!/usr/bin/env bash
# run_all_perf.sh — full dual-stack performance suite.
#
# Runs every bench family in BOTH environments (self stack + official wheel)
# using the same unmodified operator sources, then renders the merged tables
# and the headline arithmetic mean into artifacts/perf/.
#
# GPU note: captures record utilization/clocks/power next to each result.
# For exclusive-GPU formal captures, confirm `nvidia-smi dmon -c 3` shows
# ~0% SM utilization from other tenants first, then run this script.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SELF=.venv-self/bin/python
OFF=.venv-reference/bin/python
for py in "$SELF" "$OFF"; do
    if [[ ! -x $py ]]; then
        echo "error: $py not found — run tools/make_envs.sh first (docs/BUILD.md)" >&2
        exit 2
    fi
done

BENCHES=(
    bench_elementwise.py
    bench_dense_gemm.py
    bench_blockscaled.py
    bench_flashinfer_norm.py
    bench_flashinfer_b12x.py
)

fail=0
for bench in "${BENCHES[@]}"; do
    echo "=== [$bench | self] ==="
    PYTHONPATH="$ROOT/python:$ROOT/python/cutlass_compat" \
        "$SELF" "tools/perf/$bench" || fail=1
    echo
    echo "=== [$bench | official] ==="
    "$OFF" "tools/perf/$bench" || fail=1
    echo
done

echo "=== rendering tables ==="
"$SELF" tools/perf/render_tables.py || fail=1

if [[ $fail -ne 0 ]]; then
    echo "RESULT: some benches failed — inspect output above"
fi
exit $fail
