
#ifndef PYLIR_CONVERSION_PASSES
#define PYLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertPylirToLLVM : Pass<"convert-pylir-to-llvm", "::mlir::ModuleOp"> {
    let summary = "Convert Pylir dialects to LLVM dialect";
    let constructor = "::pylir::createConvertPylirToLLVMPass()";
    let dependentDialects = ["::mlir::LLVM::LLVMDialect",
                             "::pylir::Mem::PylirMemDialect",
                             "::pylir::Py::PylirPyDialect"];

    let options = [
        Option<"m_targetTripleCLI", "target-triple", "std::string",
                    /*default=*/"LLVM_DEFAULT_TARGET_TRIPLE", "LLVM target triple">,
        Option<"m_dataLayoutCLI", "data-layout", "std::string", /*default=*/"\"\"", "LLVM data layout">
    ];
}

def ConvertPylirPyToPylirMem : Pass<"convert-pylirPy-to-pylirMem", "::mlir::ModuleOp"> {
    let summary = "Convert Pylir Py dialect to Pylir Mem dialect";
    let constructor = "::pylir::createConvertPylirPyToPylirMemPass()";
    let dependentDialects = ["::pylir::Mem::PylirMemDialect", "::mlir::arith::ArithmeticDialect"];
}

#endif
