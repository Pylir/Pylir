
#pragma once

// Here due to the lack of a better place
// Does postprocessing that isn't possible in MLIR because its LLVM Dialect is missing stuff. Once those are added
// up stream this should hopefully not be necessary anymore in the future

#include <llvm/IR/Module.h>

namespace pylir
{
void postProcessLLVMModule(llvm::Module& module);
}
