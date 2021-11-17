
#ifndef PYLIR_BASE_TABLEGEN
#define PYLIR_BASE_TABLEGEN

include "mlir/IR/OpBase.td"

def PylirMem_Dialect : Dialect {
    let name = "pyMem";
    let summary = "A dialect representing python programs";
    let description = [{
        TODO
    }];
    let cppNamespace = "::pylir::Mem";
}

#endif
