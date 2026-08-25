#ifndef SELF_CUTE_PIPELINE_OPS_H
#define SELF_CUTE_PIPELINE_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::selfcute::pipeline {

class PipelineDialect : public Dialect {
public:
  explicit PipelineDialect(MLIRContext *ctx);
  static StringRef getDialectNamespace() { return "selfcute_pipeline"; }
};

} // namespace mlir::selfcute::pipeline

#define GET_OP_CLASSES
#include "PipelineOps.h.inc"

#endif // SELF_CUTE_PIPELINE_OPS_H
