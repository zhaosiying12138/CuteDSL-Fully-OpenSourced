#!/usr/bin/env bash
# run_correctness.sh — one-shot correctness regression for the self stack.
#
# Runs, in order:
#   1. host/compiler tests        (pytest -m "not sm120")
#   2. on-GPU release-gate tests  (pytest -m sm120)   [needs the 5090]
#   3. selfcute dialect LIT checks (ninja check-selfcute-lit)
#
# Expected on a healthy tree: 261 pytest cases pass, 0 fail (185 host/compiler
# plus 76 sm120), and all LIT checks are green.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv-self/bin/python
if [[ ! -x $PY ]]; then
    echo "error: $PY not found — run tools/make_envs.sh first (see docs/BUILD.md)" >&2
    exit 2
fi

fail=0

echo "=== [1/3] host / compiler tests (pytest -m 'not sm120') ==="
$PY -m pytest -m "not sm120" -q || fail=1

echo
echo "=== [2/3] on-GPU tests (pytest -m sm120) ==="
$PY -m pytest -m sm120 -q || fail=1

echo
echo "=== [3/3] selfcute dialect LIT (ninja check-selfcute-lit) ==="
ninja -C build-selfcute check-selfcute-lit || fail=1

echo
if [[ $fail -ne 0 ]]; then
    echo "RESULT: FAILURES — see output above"
else
    echo "RESULT: all correctness gates green"
fi
exit $fail
