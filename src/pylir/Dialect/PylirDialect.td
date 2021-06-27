
#ifndef PYLIR_DIALECT_TABLEGEN
#define PYLIR_DIALECT_TABLEGEN

include "mlir/IR/OpBase.td"

def Pylir_Dialect : Dialect {
    let name = "pylir";
    let summary = "A dialect representing python programs";
    let description = [{
        TODO
    }];
    let cppNamespace = "::pylir::Dialect";
}

class Pylir_Op<string mneomic, list<OpTrait> traits = []> :
    Op<Pylir_Dialect, mneomic, traits>;

class Pylir_Type<string name,string baseCppClass = "::mlir::Type"> : TypeDef<Pylir_Dialect, name,baseCppClass>;



#endif
