// selfcute.pipeline dialect — async pipeline states, mbarrier protocol,
// producer/consumer stage tracking (PipelineTmaAsync & friends).
// Skeleton (M1): dialect + seed ops, parse/print/verify only.

#ifndef SELF_CUTE_PIPELINE_OPS
#define SELF_CUTE_PIPELINE_OPS

include "mlir/IR/OpBase.td"

def SelfCute_Pipeline_Dialect : Dialect {
  let name = "selfcute_pipeline";
  let summary = "async pipeline / mbarrier producer-consumer protocol ops";
  let description = [{
    The selfcute.pipeline dialect models multi-stage async pipelines:
    per-stage acquire/release with phase parity, mbarrier arrive/wait with
    token accounting, and the TMA producer/consumer protocol used by the
    SM120 flagship kernels (PipelineTmaAsync / PipelineTmaStore).
  }];
  let cppNamespace = "::mlir::selfcute::pipeline";
}

class SelfCute_Pipeline_Op<string mnemonic, list<Trait> traits = []> :
    Op<SelfCute_Pipeline_Dialect, mnemonic, traits>;

def SelfCute_Pipeline_AcquireOp : SelfCute_Pipeline_Op<"acquire"> {
  let summary = "Consumer acquires pipeline stage `stage` at phase `phase`";
  let description = [{
    Blocks the executing warp until the producer's mbarrier for (stage,
    phase) has completed its arrival count. Lowered to mbarrier.try_wait
    parity loops.
  }];
  let arguments = (ins AnyType:$state, I32:$stage, I1:$phase);
  let results = (outs AnyType:$new_state);
}

def SelfCute_Pipeline_ReleaseOp : SelfCute_Pipeline_Op<"release"> {
  let summary = "Producer releases pipeline stage `stage` at phase `phase`";
  let description = [{
    Signals consumers that (stage, phase) is reusable: expect-tx / arrive
    on the stage's mbarrier. Lowered to mbarrier.arrive (+ expect_tx when
    paired with a TMA issue).
  }];
  let arguments = (ins AnyType:$state, I32:$stage, I1:$phase);
  let results = (outs AnyType:$new_state);
}

#endif // SELF_CUTE_PIPELINE_OPS
