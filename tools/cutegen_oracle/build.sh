#!/usr/bin/env bash
# build.sh — build the in-process cutegen oracle binding (nanobind).
#
# Produces build-oracle/_cutegen_oracle<...>.so, loaded by
# python/self_cutedsl/object_model/cutegen_binding.py. Header-only cutegen
# comes from the vendored cutlass_compiler; nanobind from .venv-self.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PY=.venv-self/bin/python
NANOBIND_INC=$($PY -c "import nanobind, os; print(os.path.dirname(nanobind.__file__))")
CUTEGEN_INC="$ROOT/third_party/cutlass/cutlass_compiler/cutegen/include"
PYINC=$($PY -c "import sysconfig; print(sysconfig.get_paths()['include'])")

mkdir -p build-oracle
g++ -std=c++20 -O2 -fPIC -shared \
    -I "$CUTEGEN_INC" -I "$NANOBIND_INC/include" -I "$NANOBIND_INC/ext/robin_map/include" -I "$PYINC" \
    tools/cutegen_oracle/binding.cpp \
    "$NANOBIND_INC/src/nb_combined.cpp" \
    -o build-oracle/_cutegen_oracle.so

$PY - <<'EOF'
import sys
sys.path.insert(0, "build-oracle")
import _cutegen_oracle as O
r = O.selfcheck()
assert r == "(32,4):(1,8)", r
d = O.count_dynamics("(256,?):(?,1)")
assert d == 2, d
print("[cutegen_oracle] selfcheck OK:", r, "| dynamics(256,?):(?,1) =", d)
EOF
echo "[cutegen_oracle] built build-oracle/_cutegen_oracle.so"
