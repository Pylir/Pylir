// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYLIRHIR_TRANSFORMS_PASSES
#define PYLIRHIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def FuncOutliningPass : Pass<"pylir-func-outlining", "mlir::ModuleOp"> {
  let dependentDialects = ["::pylir::HIR::PylirHIRDialect",
                           "::pylir::Py::PylirPyDialect",
                           "::mlir::arith::ArithDialect"];

  let description = [{
    Outlines all `pyHIR.func` ops and transforms them to `pyHIR.globalFunc`
    ops and function object creation ops.
  }];
}

def ClassBodyOutliningPass : Pass<"pylir-class-body-outlining"> {
  let dependentDialects = ["::pylir::Py::PylirPyDialect"];

  let description = [{
    Converts all `pyHIR.class` ops to by outlining the bodies to `pyHIR.func`
    ops and replacing them with `pyHIR.buildClass` ops.
  }];
}

#endif
