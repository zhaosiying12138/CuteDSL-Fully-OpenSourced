#ifndef SELF_CUTE_REGISTRATION_H
#define SELF_CUTE_REGISTRATION_H

#include "mlir/IR/DialectRegistry.h"

namespace mlir::selfcute {

void registerSelfCuteDialects(DialectRegistry &registry);

} // namespace mlir::selfcute

#endif // SELF_CUTE_REGISTRATION_H
