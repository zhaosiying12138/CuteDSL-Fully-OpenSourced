#ifndef SELF_CUTE_SM120_OPS_H
#define SELF_CUTE_SM120_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::selfcute::sm120 {

class SM120Dialect : public Dialect {
public:
  explicit SM120Dialect(MLIRContext *ctx);
  static StringRef getDialectNamespace() { return "selfcute_sm120"; }
};

} // namespace mlir::selfcute::sm120

#define GET_OP_CLASSES
#include "SM120Ops.h.inc"

#endif // SELF_CUTE_SM120_OPS_H
