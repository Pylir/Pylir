
#ifndef PYLIR_ATTRIBUTES_TABLEGEN
#define PYLIR_ATTRIBUTES_TABLEGEN

include "pylir/Optimizer/PylirMem/IR/PylirMemBase.td"

class Pylir_Attr<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Attribute"> :
    AttrDef<Pylir_Dialect, name, traits, baseCppClass>;

#endif
