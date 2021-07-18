
#ifndef PYLIR_TO_LLVM_PATTERNS
#define PYLIR_TO_LLVM_PATTERNS

include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOps.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "pylir/Optimizer/Dialect/PylirOps.td"

def : Pat<(Pylir_HandleOfOp $global),(LLVM_AddressOfOp $global)>;

#endif
