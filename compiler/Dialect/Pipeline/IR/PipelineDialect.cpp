#include "SelfCute/Pipeline/IR/PipelineOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::selfcute::pipeline;

PipelineDialect::PipelineDialect(MLIRContext *ctx) : Dialect(getDialectNamespace(), ctx, TypeID::get<PipelineDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "PipelineOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "PipelineOps.cpp.inc"
