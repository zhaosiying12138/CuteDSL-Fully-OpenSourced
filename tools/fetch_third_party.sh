#!/usr/bin/env bash
# Fetch pinned third-party sources that are too large to vendor into the repo.
#
# Vendored (committed, BSD-3): third_party/cutlass/{cutlass_compiler, examples}
# Fetched by this script (gitignored tree, pinned by compat/sm120_toolchain.lock.yaml):
#   third_party/flashinfer-src   @ flashinfer_commit
#
# Usage: tools/fetch_third_party.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FLASHINFER_COMMIT="$(grep -m1 '^flashinfer_commit:' compat/sm120_toolchain.lock.yaml | awk '{print $2}')"

if [ -z "$FLASHINFER_COMMIT" ]; then
    echo "[fetch_third_party] flashinfer_commit missing from compat/sm120_toolchain.lock.yaml" >&2
    exit 1
fi

mkdir -p third_party/flashinfer-src
cd third_party/flashinfer-src
if [ ! -d .git ]; then
    # Prefer a shared clone from a local flashinfer checkout when available
    # (object reuse, no network); fall back to a shallow network fetch.
    if [ -d /home/zhaosiying/codebase/flashinfer/.git ]; then
        git clone --shared --no-checkout /home/zhaosiying/codebase/flashinfer .
    else
        git init
        git remote add origin git@github.com:flashinfer-ai/flashinfer.git
    fi
fi
if ! git remote get-url origin >/dev/null 2>&1; then
    # github.com SSH (port 443) is the reliable transport on this network.
    git remote add origin git@github.com:flashinfer-ai/flashinfer.git
fi
if ! git cat-file -e "${FLASHINFER_COMMIT}^{commit}" 2>/dev/null; then
    git fetch --depth 1 origin "${FLASHINFER_COMMIT}"
fi
git checkout --detach "${FLASHINFER_COMMIT}"
echo "[fetch_third_party] flashinfer @ ${FLASHINFER_COMMIT} ready under third_party/flashinfer-src/"
