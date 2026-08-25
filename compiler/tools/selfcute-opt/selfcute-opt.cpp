// selfcute-opt: MLIR tool with the three selfcute dialects registered
// alongside the public cute dialect and the base facade, for parse/print/
// verify round-trips and (later) selfcute pass pipelines.
#include "SelfCute/Registration/RegisterSelfCute.h"

#include "base/Registration/Registration.h"
#include "cute_ir/Registration/Registration.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;

  // Public layers (BSD cutlass_compiler).
  cutlass_compiler::cute::registerCuteDialects(registry);
  cutlass_compiler::cute::registerCutePasses();
  cutlass_compiler::base::registerBaseDialects(registry);
  cutlass_compiler::base::registerBasePasses();

  // Self dialects.
  selfcute::registerSelfCuteDialects(registry);

  registerTransformsPasses();
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "selfcute-opt: cute + base + selfcute dialects\n", registry));
}
