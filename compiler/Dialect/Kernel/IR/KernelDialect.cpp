#include "SelfCute/Kernel/IR/KernelOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::selfcute::kernel;

KernelDialect::KernelDialect(MLIRContext *ctx) : Dialect(getDialectNamespace(), ctx, TypeID::get<KernelDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "KernelOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "KernelOps.cpp.inc"
