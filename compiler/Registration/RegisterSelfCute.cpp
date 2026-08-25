#include "SelfCute/Registration/RegisterSelfCute.h"

#include "SelfCute/Kernel/IR/KernelOps.h"
#include "SelfCute/Pipeline/IR/PipelineOps.h"
#include "SelfCute/SM120/IR/SM120Ops.h"

using namespace mlir;

void mlir::selfcute::registerSelfCuteDialects(DialectRegistry &registry) {
  registry.insert<selfcute::kernel::KernelDialect>();
  registry.insert<selfcute::pipeline::PipelineDialect>();
  registry.insert<selfcute::sm120::SM120Dialect>();
}
