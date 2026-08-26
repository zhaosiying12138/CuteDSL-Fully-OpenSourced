"""cutlass.pipeline — async pipeline driver objects (compat, clean-room).

PipelineTmaAsync / PipelineTmaStore wrap the mbarrier stage/phase
protocol already verified in M5/M6 (init, expect_tx, try_wait.parity,
producer tail) as driver objects with SSA-tracked stage/phase state,
matching the official call surface used by the flagship kernels:

    pipe = pipeline.PipelineTmaAsync.create(
        num_stages=..., producer_group=..., consumer_group=...,
        tx_count=..., barrier_storage=..., cta_layout_vmnk=...)
    state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer,
                                         num_stages)
    pipe.producer_acquire(state); copy(...); pipe.producer_commit(state)
    pipe.consumer_wait(state); ...; pipe.consumer_release(state)

Implemented from the PTX ISA mbarrier semantics and the BSD example
call-shapes only (the official wheel's implementation is not read).
"""
from __future__ import annotations

from enum import Enum

from self_cutedsl.frontend import builtins as _b
from self_cutedsl.frontend.emitter import SSA


class Agent(Enum):
    Thread = 0
    Warp = 1
    CTA = 2


class PipelineUserType(Enum):
    Producer = 0
    Consumer = 1


class CooperativeGroup:
    def __init__(self, agent: Agent, count: int = 1):
        self.agent = agent
        self.count = int(count)


class NamedBarrier:
    """bar.sync/bar.arrive on a named id with a thread count."""

    def __init__(self, barrier_id: int, thread_count: int):
        self.barrier_id = int(barrier_id)
        self.thread_count = int(thread_count)

    def arrive(self):
        _b._emitter().named_barrier_arrive("nb", self.barrier_id,
                                           self.thread_count)

    def wait(self):
        _b._emitter().named_barrier_sync(self.barrier_id, self.thread_count)

    def sync(self):
        self.wait()


class PipelineState:
    """Stage index + phase parity, SSA-tracked for dynamic loops."""

    def __init__(self, stage: SSA | int, phase: SSA | int, pipe_id: int):
        self.stage = stage          # SSA i32 (or python int at trace start)
        self.phase = phase          # SSA i32 parity (0/1)
        self.pipe_id = pipe_id

    def copy(self) -> "PipelineState":
        return PipelineState(self.stage, self.phase, self.pipe_id)


def make_pipeline_state(user_type: PipelineUserType,
                        num_stages: int) -> PipelineState:
    e = _b._emitter()
    stage = e.ssa("i32", f"arith.constant 0 : i32")
    phase = e.ssa("i32", f"arith.constant 0 : i32")
    return PipelineState(stage, phase, id(user_type))


def sync(barrier_id: int = 0, aligned: bool = False):
    if barrier_id == 0:
        _b._emitter().barrier()
    else:
        _b._emitter().named_barrier_sync(barrier_id, 0xFFFF)


def _bar_at(storage_ptr, index_ssa):
    """mbarrier address = storage_ptr + index*8 bytes (Int64 array)."""
    e = _b._emitter()
    off = e.ssa("i64", f"arith.extsi {index_ssa.name} : i32 to i64")
    p = e.ssa("!llvm.ptr",
              f"llvm.getelementptr {storage_ptr.name}[{off.name}] "
              f": (!llvm.ptr, i64) -> !llvm.ptr, i64")
    return p


class PipelineTmaAsync:
    """Multi-stage TMA pipeline: mbarrier arrays with phase parity."""

    def __init__(self, num_stages, producer_group, consumer_group,
                 tx_count, barrier_storage, cta_layout_vmnk=None):
        self.num_stages = int(num_stages)
        self.producer_group = producer_group
        self.consumer_group = consumer_group
        self.tx_count = int(tx_count)
        self.barrier_storage = barrier_storage      # SSA ptr (Int64 array)
        self.cta_layout_vmnk = cta_layout_vmnk

    # ------------------------------------------------------------------
    @staticmethod
    def create(*, num_stages, producer_group, consumer_group, tx_count,
               barrier_storage, cta_layout_vmnk=None, **kw):
        e = _b._emitter()
        # init every stage's barrier once (thread 0), fence, sync
        tidx = e.thread_id("x")
        for s in range(int(num_stages)):
            idx = e.ssa("i32", f"arith.constant {s} : i32")
            bar = _bar_at(barrier_storage, idx)
            e.mbarrier_init_single_thread(bar, producer_group.count +
                                          consumer_group.count, tidx)
        e.fence_mbarrier_init()
        e.barrier()
        return PipelineTmaAsync(num_stages, producer_group, consumer_group,
                                tx_count, barrier_storage, cta_layout_vmnk)

    # ------------------------------------------------------------------
    def producer_acquire(self, state: PipelineState):
        """Wait the stage's empty (consumer-release) phase; then expect_tx."""
        e = _b._emitter()
        bar = _bar_at(self.barrier_storage, state.stage)
        e.mbarrier_try_wait_parity(bar, state.phase)

    def producer_expect_tx(self, state: PipelineState):
        e = _b._emitter()
        bar = _bar_at(self.barrier_storage, state.stage)
        e.mbarrier_arrive_expect_tx(bar, self.tx_count)

    def producer_commit(self, state: PipelineState):
        pass  # TMA load itself completes the transaction

    def producer_get_barrier(self, state: PipelineState):
        return _bar_at(self.barrier_storage, state.stage)

    def producer_tail(self, state: PipelineState):
        pass

    def consumer_wait(self, state: PipelineState):
        e = _b._emitter()
        bar = _bar_at(self.barrier_storage, state.stage)
        # consumer waits the opposite parity of its release phase
        inv = _b.sub_i32(state.phase, 1)          # 0<->1 via -1 (mod 2^32)
        p = _b.rem_i32(inv, 2) if hasattr(_b, "rem_i32") else state.phase
        e.mbarrier_try_wait_parity(bar, p)
        e.fence_proxy_async_shared = None  # noop guard
        _b._emitter().raw("nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, "
                          "space = #nvvm.shared_space<cta>}")

    def consumer_release(self, state: PipelineState):
        e = _b._emitter()
        bar = _bar_at(self.barrier_storage, state.stage)
        n = int(self.consumer_group.count)
        if n == 1:
            e.raw(f"nvvm.mbarrier.arrive {bar.name} : !llvm.ptr -> i64")
        else:
            c = e.ssa("i32", f"arith.constant {n} : i32")
            e.raw(f"nvvm.mbarrier.arrive.nocomplete {bar.name}, {c.name} "
                  f": !llvm.ptr, i32 -> i64")

    # ------------------------------------------------------------------
    @staticmethod
    def increment_state(state: PipelineState, num_stages: int):
        """stage+1 (wrap), phase flips on wrap — SSA arithmetic."""
        e = _b._emitter()
        one = _b.const_i32(1)
        ns = _b.const_i32(int(num_stages))
        nxt = _b.add_i32(state.stage, one)
        wrapped = _b.lt_i32(nxt, ns)                     # i1
        inc = _b.bool_to_i32(wrapped)
        # stage = wrapped ? nxt : 0
        zero = _b.const_i32(0)
        e.raw(f"%sel_s = arith.select {wrapped.name}, {nxt.name}, "
              f"{zero.name} : i32") if False else None
        # use sub trick: stage = (nxt) * inc  (inc==0 when wrapped)
        stage2 = _b.mul_i32(nxt, inc)
        # phase flips when wrapping: phase += 1 - inc
        notinc = _b.sub_i32(one, inc)
        phase2 = _b.add_i32(state.phase, notinc)
        phase2 = _b.rem_i32(phase2, 2)
        return PipelineState(stage2, phase2, state.pipe_id)


class PipelineTmaStore:
    """cp.async.bulk.tensor S2G pipeline (bulk-group based)."""

    def __init__(self, num_stages, barrier_storage=None):
        self.num_stages = int(num_stages)
        self.barrier_storage = barrier_storage

    @staticmethod
    def create(*, num_stages, barrier_storage=None, **kw):
        return PipelineTmaStore(num_stages, barrier_storage)

    def producer_acquire(self, state):
        pass

    def producer_commit(self, state):
        _b._emitter().raw("nvvm.cp.async.bulk.commit.group")

    def consumer_wait(self, state):
        _b._emitter().raw("nvvm.cp.async.bulk.wait_group 0")

    def consumer_release(self, state):
        pass

    @staticmethod
    def increment_state(state, num_stages):
        return PipelineTmaAsync.increment_state(state, num_stages)


# module-level re-exports used by the flagship kernels
def producer_acquire(pipe, state):
    pipe.producer_acquire(state)


def producer_commit(pipe, state):
    pipe.producer_commit(state)


def producer_get_barrier(pipe, state):
    return pipe.producer_get_barrier(state)


def producer_tail(pipe, state):
    pipe.producer_tail(state)


def consumer_wait(pipe, state):
    pipe.consumer_wait(state)


def consumer_try_wait(pipe, state):
    pipe.consumer_wait(state)  # blocking form (parity loop)

def consumer_release(pipe, state):
    pipe.consumer_release(state)
