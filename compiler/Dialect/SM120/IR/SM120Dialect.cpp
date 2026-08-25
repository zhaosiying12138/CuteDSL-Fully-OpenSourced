#include "SelfCute/SM120/IR/SM120Ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::selfcute::sm120;

SM120Dialect::SM120Dialect(MLIRContext *ctx) : Dialect(getDialectNamespace(), ctx, TypeID::get<SM120Dialect>()) {
  addOperations<
#define GET_OP_LIST
#include "SM120Ops.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "SM120Ops.cpp.inc"
