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

    def __init__(self, barrier_id: int, thread_count: int = None,
                 num_threads: int = None):
        self.barrier_id = int(barrier_id)
        self.thread_count = int(thread_count or num_threads or 0)

    def arrive_and_wait(self):
        self.sync()

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

    def __snapshot__(self):
        return (
            self.stage,
            self.phase,
            self.pipe_id,
            getattr(self, "num_stages", None),
            getattr(self, "_count", 0),
        )

    def __restore__(self, snapshot):
        self.stage, self.phase, self.pipe_id, num_stages, self._count = snapshot
        if num_stages is not None:
            self.num_stages = num_stages

    def advance(self, num_stages: int = None):
        """stage+1 wrap / phase flip — SSA select arithmetic (in place)."""
        e = _b._emitter()
        ns = int(num_stages or getattr(self, "num_stages", 2))
        one = _b.const_i32(1)
        cns = _b.const_i32(ns)
        nxt = _b.add_i32(self.stage, one)
        wrapped = _b.lt_i32(nxt, cns)
        inc = _b.bool_to_i32(wrapped)
        self.stage = _b.mul_i32(nxt, inc)
        notinc = _b.sub_i32(one, inc)
        self.phase = _b.rem_i32(_b.add_i32(self.phase, notinc), 2)
        self._count = getattr(self, "_count", 0) + 1

    def reset_count(self):
        self._count = 0

    @property
    def count(self):
        return getattr(self, "_count", 0)

    @property
    def index(self):
        return self.stage


def make_pipeline_state(user_type: PipelineUserType,
                        num_stages: int) -> PipelineState:
    e = _b._emitter()
    stage = e.ssa("i32", "arith.constant 0 : i32")
    # producer waits the EMPTY parity: a fresh barrier passes parity-1
    # immediately (previous-phase convention); consumer waits parity 0
    ph = 1 if user_type is PipelineUserType.Producer else 0
    phase = e.ssa("i32", f"arith.constant {ph} : i32")
    st = PipelineState(stage, phase, id(user_type))
    st.num_stages = int(num_stages)
    st._count = 0
    return st


def sync(barrier_id: int = 0, aligned: bool = False):
    if barrier_id == 0:
        _b._emitter().barrier()
    else:
        # count-free form: all CTA threads participate (warp-multiple)
        _b._emitter().raw(f"nvvm.inline_ptx \"bar.sync {int(barrier_id)};\"")


def _bar_at(storage_ptr, index_ssa):
    """mbarrier address = storage_ptr + index*8 bytes (Int64 smem array);
    stays in addrspace(3) — TMA ops consume !llvm.ptr<3> barriers."""
    e = _b._emitter()
    off = e.ssa("i64", f"arith.extsi {index_ssa.name} : i32 to i64")
    p = e.ssa("!llvm.ptr<3>",
              f"llvm.getelementptr {storage_ptr.name}[{off.name}] "
              f": (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i64")
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

    def __snapshot__(self):
        return getattr(self, "_uses", 0)

    def __restore__(self, snapshot):
        self._uses = snapshot

    # ------------------------------------------------------------------
    @staticmethod
    def create(*, num_stages, producer_group, consumer_group, tx_count,
               barrier_storage, cta_layout_vmnk=None, **kw):
        e = _b._emitter()
        # two-barrier protocol (the 2*num_stages storage exists for this):
        # FULL[s] count=1 (expect_tx arrival + tx bytes) = data ready;
        # EMPTY[s] count = consumer warp count — lane0 of every consumer
        # warp arrives, so a stage frees only after ALL warps' register
        # loads are past it (one rep thread would race the others)
        tidx = e.thread_id("x")
        ns = int(num_stages)
        nw = int(getattr(consumer_group, "count", 0) or 0) or 1
        for s in range(ns):
            idx = e.ssa("i32", f"arith.constant {s} : i32")
            bar = _bar_at(barrier_storage, idx)
            e.mbarrier_init_single_thread(bar, 1, tidx)
        for s in range(ns, 2 * ns):
            idx = e.ssa("i32", f"arith.constant {s} : i32")
            bar = _bar_at(barrier_storage, idx)
            e.mbarrier_init_single_thread(bar, nw, tidx)
        e.fence_mbarrier_init()
        e.barrier()
        pipe = PipelineTmaAsync(num_stages, producer_group, consumer_group,
                                tx_count, barrier_storage, cta_layout_vmnk)
        pipe._uses = 0         # trace-time acquire counter (stage cycling)
        return pipe

    # ------------------------------------------------------------------
    def producer_acquire(self, state: PipelineState):
        """Fresh stages (first num_stages acquires) need no empty-wait;
        reused stages wait EMPTY[stage] at the previous-use parity, then
        arm expect_tx on FULL[stage] (leader-elected)."""
        e = _b._emitter()
        u = getattr(self, "_uses", 0)
        ns = self.num_stages
        n = u // ns                  # which use of this stage this is
        if n > 0:
            empty_idx = _b.add_i32(state.stage, _b.const_i32(ns))
            bar = _bar_at(self.barrier_storage, empty_idx)
            par = _b.const_i32((n - 1) % 2)
            e.mbarrier_try_wait_parity(bar, par)
        self._uses = u + 1
        pred = _b._lane0_predicate()
        e.open_if(pred)
        bar2 = _bar_at(self.barrier_storage, state.stage)
        e.mbarrier_arrive_expect_tx(bar2, self.tx_count)
        e.close_if()

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

    def consumer_wait(self, state: PipelineState, peek_status=None):
        """Wait parity == state.phase + async-proxy fence (verified M5/M6
        protocol); peek_status is an optimization hint — the wait itself is
        authoritative and idempotent on an already-completed phase."""
        e = _b._emitter()
        bar = _bar_at(self.barrier_storage, state.stage)
        e.mbarrier_try_wait_parity(bar, state.phase)
        e.fence_proxy_async_shared()

    def consumer_try_wait(self, state: PipelineState):
        """Peek: conservative constant-true (the following consumer_wait
        performs the real blocking wait)."""
        e = _b._emitter()
        return e.ssa("i1", "arith.constant 1 : i1")

    def consumer_release(self, state: PipelineState):
        """Lane0 of EVERY consumer warp arrives on EMPTY[stage] (count =
        warp count): the stage frees only when all warps have passed."""
        e = _b._emitter()
        pred = _b._lane0_predicate()
        e.open_if(pred)
        empty_idx = _b.add_i32(state.stage, _b.const_i32(self.num_stages))
        bar = _bar_at(self.barrier_storage, empty_idx)
        e.raw(f"nvvm.mbarrier.arrive {bar.name} : !llvm.ptr<3> -> i64")
        e.close_if()

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

    def producer_acquire(self, state=None):
        pass

    def producer_commit(self, state=None):
        _b._emitter().raw("nvvm.cp.async.bulk.commit.group")

    def producer_tail(self, state=None):
        _b._emitter().raw("nvvm.cp.async.bulk.wait_group 0")

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
