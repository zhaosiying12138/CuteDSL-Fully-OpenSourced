#ifndef SELF_CUTE_KERNEL_OPS_H
#define SELF_CUTE_KERNEL_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::selfcute::kernel {

class KernelDialect : public Dialect {
public:
  explicit KernelDialect(MLIRContext *ctx);
  static StringRef getDialectNamespace() { return "selfcute_kernel"; }
};

} // namespace mlir::selfcute::kernel

#define GET_OP_CLASSES
#include "KernelOps.h.inc"

#endif // SELF_CUTE_KERNEL_OPS_H
