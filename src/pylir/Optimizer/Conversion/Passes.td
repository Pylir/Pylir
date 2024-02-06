// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIR_CONVERSION_PASSES
#define PYLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertPylirToLLVMPass : Pass<"convert-pylir-to-llvm", "::mlir::ModuleOp"> {
  let summary = "Convert Pylir dialects to LLVM dialect";

  let dependentDialects = ["::mlir::LLVM::LLVMDialect",
               "::pylir::Mem::PylirMemDialect",
               "::pylir::Py::PylirPyDialect"];

  let options = [
    Option<"m_targetTripleCLI", "target-triple", "std::string",
          /*default=*/"LLVM_DEFAULT_TARGET_TRIPLE", "LLVM target triple">,
    Option<"m_dataLayoutCLI", "data-layout", "std::string", /*default=*/"\"\"",
      "LLVM data layout">
  ];
}

def ConvertPylirPyToPylirMemPass : Pass<"convert-pylirPy-to-pylirMem",
  "::mlir::ModuleOp"> {
  let summary = "Convert Pylir Py dialect to Pylir Mem dialect";

  let dependentDialects = [
    "::pylir::Mem::PylirMemDialect",
    "::mlir::arith::ArithDialect"
  ];
}

def ConvertPylirHIRToPylirPyPass : Pass<"convert-pylirHIR-to-pylirPy",
  "::mlir::ModuleOp"> {
  let summary = "Convert PylirHIR dialect to PylirPy dialect";

  let dependentDialects = [
    "::pylir::Py::PylirPyDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::cf::ControlFlowDialect",
  ];
}

#endif
