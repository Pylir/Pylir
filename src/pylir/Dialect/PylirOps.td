
#ifndef PYLIR_OPS_TABLEGEN
#define PYLIR_OPS_TABLEGEN

include "PylirDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Pylir_FooOp : Pylir_Op<"foo", [NoSideEffect,
                                   SameOperandsAndResultType]> {
    let summary = "test";
    let description = [{
        TODO
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

#endif
