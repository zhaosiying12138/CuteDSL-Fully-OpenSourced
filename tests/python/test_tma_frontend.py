"""M5 frontend slice: TMA roundtrip through @cute.jit.

Same protocol as the raw-MLIR slice, expressed with the frontend
builtins: mbarrier init/fence, producer expect_tx + TMA load, consumer
parity wait, plain SMEM read, increment, TMA store.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/cutlass_compat"))

import cutlass
import cutlass.cute as cute
from self_cutedsl.runtime.tensor_map import TensorMapRecipe, CUtensorMapView


@cute.kernel
def tma_kernel(a: cute.TmaTensor, b: cute.TmaTensor, out: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    is0 = tidx == 0

    smem = cute.make_smem_tile("fe_tile", 32)            # 4x8 f32
    bar = cute.make_mbarrier("fe_bar", 1)
    cute.sync_threads()

    if is0:
        cute.mbarrier_arrive_expect_tx(bar, 128)
        cute.tma_load(a, smem, bar, [0, 0])
    cute.mbarrier_try_wait_parity(bar, 0)
    cute.sync_threads()

    # golden 1: plain read of the TMA-loaded tile
    out[tidx] = smem[tidx]

    # golden 2: increment, TMA-store to b
    one = 100.0
    smem[tidx] = out[tidx] + one
    cute.sync_threads()
    if is0:
        cute.tma_store(b, smem, [0, 0])
    cute.sync_threads()


@cute.jit
def run_tma(a, b, out):
    tma_kernel(a, b, out).launch(grid=(1, 1, 1), block=(32, 1, 1))


@pytest.mark.sm120
def test_tma_frontend_roundtrip():
    cutlass.cuda.initialize_cuda_context()
    src = torch.arange(32, device="cuda", dtype=torch.float32).reshape(4, 8) / 7.0
    dst = torch.zeros(4, 8, device="cuda", dtype=torch.float32)
    plain = torch.zeros(32, device="cuda", dtype=torch.float32)

    recipe = TensorMapRecipe(dtype=torch.float32, shape=(4, 8),
                             strides_elems=(8, 1), box=(8, 4))
    tma_in = CUtensorMapView(recipe, src)
    tma_out = CUtensorMapView(recipe, dst)

    run_tma(tma_in, tma_out, plain)
    torch.cuda.synchronize()

    torch.testing.assert_close(plain.reshape(4, 8), src)
    torch.testing.assert_close(dst, src + 100.0)
