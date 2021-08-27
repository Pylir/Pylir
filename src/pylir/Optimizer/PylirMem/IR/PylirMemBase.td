
#ifndef PYLIR_BASE_TABLEGEN
#define PYLIR_BASE_TABLEGEN

include "mlir/IR/OpBase.td"

def Pylir_Dialect : Dialect {
    let name = "pylir.mem";
    let summary = "A dialect representing python programs";
    let description = [{
        TODO
    }];
    let cppNamespace = "::pylir::Mem";
}

#endif
