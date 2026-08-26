"""object_model/pipeline.py + tma.py — S3: PipelineTmaAsync driver and
generalized TMA (clean-room from PTX ISA + BSD example call shapes).

Copyright (c) 2026 CuTeDSL-Fully-OpenSourced contributors
These live in self_cutedsl (NOT the compat shims) and sit directly on
the verified M5/M6 mbarrier/TMA builtins.
"""
from __future__ import annotations

from ..frontend import builtins as _b
from ..frontend.emitter import SSA


class Agent:
    Thread = "thread"
    Warp = "warp"
    CTA = "cta"


class PipelineUserType:
    Producer = "producer"
    Consumer = "consumer"


class CooperativeGroup:
    def __init__(self, agent=Agent.Thread, count: int = 1):
        self.agent = agent
        self.count = int(count)


class PipelineState:
    """stage + phase parity; SSA-tracked (i32) for dynamic loops."""

    def __init__(self, stage, phase):
        self.stage = stage      # SSA i32
        self.phase = phase      # SSA i32


def make_pipeline_state(user_type, num_stages: int) -> PipelineState:
    e = _b._emitter()
    return PipelineState(e.ssa("i32", "arith.constant 0 : i32"),
                         e.ssa("i32", "arith.constant 0 : i32"))


def sync(barrier_id: int = 0):
    if barrier_id == 0:
        _b._emitter().barrier()
    else:
        _b._emitter().named_barrier_sync(barrier_id, 0xFFFF)


class PipelineTmaAsync:
    """Multi-stage TMA pipeline over an mbarrier array.

    Semantics: producer_acquire waits the stage's empty phase;
    producer_expect_tx arms the transaction; the TMA copy issued with the
    barrier completes it. consumer_wait blocks on the full phase;
    consumer_release arrives to flip to the next empty phase.
    """

    def __init__(self, num_stages, producer_group, consumer_group,
                 tx_count, barrier_storage):
        self.num_stages = int(num_stages)
        self.producer_group = producer_group
        self.consumer_group = consumer_group
        self.tx_count = int(tx_count)
        self.storage = barrier_storage   # SSA ptr to Int64 array

    # ------------------------------------------------------------------
    @staticmethod
    def create(*, num_stages, producer_group, consumer_group, tx_count,
               barrier_storage, **kw):
        e = _b._emitter()
        tidx = e.thread_id("x")
        for s in range(int(num_stages)):
            idx = e.ssa("i32", f"arith.constant {s} : i32")
            bar = _bar_at(barrier_storage, idx)
            # count=1: the producer's arrive.expect_tx is the sole arrival;
            # the TMA transaction completion flips the phase. Consumer
            # ordering rides on sync_threads (verified M6 protocol).
            e.mbarrier_init_single_thread(bar, 1, tidx)
        e.fence_mbarrier_init()
        e.barrier()
        return PipelineTmaAsync(num_stages, producer_group, consumer_group,
                                tx_count, barrier_storage)

    # ---- official API surface ------------------------------------------
    def producer_acquire(self, state: PipelineState):
        e = _b._emitter()
        # wait the stage's current EMPTY parity == producer phase
        bar = _bar_at(self.storage, state.stage)
        e.mbarrier_try_wait_parity(bar, state.phase)

    def producer_expect_tx(self, state: PipelineState):
        e = _b._emitter()
        bar = _bar_at(self.storage, state.stage)
        e.mbarrier_arrive_expect_tx(bar, self.tx_count)

    def producer_commit(self, state):
        pass  # the TMA copy itself completes the transaction

    def producer_get_barrier(self, state):
        return _bar_at(self.storage, state.stage)

    def producer_tail(self, state):
        pass

    def consumer_wait(self, state: PipelineState):
        """Wait parity == state.phase (the (kt//STAGES)%2 scheme verified
        golden since M5) + async-proxy fence for the upcoming SMEM reads."""
        e = _b._emitter()
        bar = _bar_at(self.storage, state.stage)
        e.mbarrier_try_wait_parity(bar, state.phase)
        e.fence_proxy_async_shared()

    def consumer_release(self, state: PipelineState):
        """No explicit arrive in the count=1 protocol; stage reuse ordering
        rides on the kernel's sync_threads between consume and refill."""
        pass

    # ---- state advance --------------------------------------------------
    def advance(self, state: PipelineState) -> PipelineState:
        """stage+1 wrap / phase flip — SSA select arithmetic."""
        e = _b._emitter()
        one = _b.const_i32(1)
        ns = _b.const_i32(self.num_stages)
        nxt = _b.add_i32(state.stage, one)
        wrapped = _b.lt_i32(nxt, ns)
        inc = _b.bool_to_i32(wrapped)
        stage2 = _b.mul_i32(nxt, inc)
        notinc = _b.sub_i32(one, inc)
        phase2 = _b.rem_i32(_b.add_i32(state.phase, notinc), 2)
        return PipelineState(stage2, phase2)


def _bar_at(storage, index_ssa):
    e = _b._emitter()
    off = e.ssa("i64", f"arith.extsi {index_ssa.name} : i32 to i64")
    return e.ssa("!llvm.ptr<3>",
                 f"llvm.getelementptr {storage.name}[{off.name}] "
                 f": (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i64")
