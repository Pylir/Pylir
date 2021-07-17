
#ifndef PYLIR_TO_LLVM_PATTERNS
#define PYLIR_TO_LLVM_PATTERNS

include "mlir/IR/OpBase.td"
include "pylir/Optimizer/Dialect/PylirOps.td"
include "mlir/Dialect/LLVMIR/LLVMOps.td"

def : Pat<(Pylir_HandleOfOp $global),(LLVM_AddressOfOp $global)>;

#endif
