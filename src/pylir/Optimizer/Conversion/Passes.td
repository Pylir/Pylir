
#ifndef PYLIR_CONVERSION_PASSES
#define PYLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertPylirToLLVM : Pass<"convert-pylirMem-to-llvm","::mlir::ModuleOp"> {
    let summary = "Convert Pylir dialect to LLVM dialect";
    let constructor = "::pylir::Mem::createConvertPylirToLLVMPass()";
    let dependentDialects = ["::mlir::LLVM::LLVMDialect"];

    let options = [
        Option<"targetTriple", "target-triple", "std::string", /*default=*/"\"\"", "LLVM target triple">,
        Option<"dataLayout", "data-layout", "std::string", /*default=*/"\"\"", "LLVM data layout">
    ];
}

#endif
