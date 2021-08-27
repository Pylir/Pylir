
#ifndef PYLIR_PY_BASE_TABLEGEN
#define PYLIR_PY_BASE_TABLEGEN

include "mlir/IR/OpBase.td"

def PylirPy_Dialect : Dialect {
    let name = "pylir.py";
    let summary = "A dialect representing python programs";
    let description = [{
        TODO
    }];
    let cppNamespace = "::pylir::Py";
}

#endif
