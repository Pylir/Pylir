
#ifndef PYLIR_TYPES_TABLEGEN
#define PYLIR_TYPES_TABLEGEN

include "pylir/Optimizer/PylirMem/IR/PylirMemBase.td"
include "mlir/IR/SubElementInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class Pylir_Type<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Type"> :
    TypeDef<Pylir_Dialect, name, traits, baseCppClass>;

#endif
