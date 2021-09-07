
#ifndef PYLIR_PY_TYPES_TABLEGEN
#define PYLIR_PY_TYPES_TABLEGEN

include "pylir/Optimizer/PylirPy/IR/PylirPyBase.td"

class PylirPy_Type<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Type"> :
    TypeDef<PylirPy_Dialect, name, traits, baseCppClass>;

def PylirPy_DynamicType : PylirPy_Type<"Dynamic"> {
    let mnemonic = "dynamic";
    let summary = "dynamic type";
}

def PylirPy_HandleType : PylirPy_Type<"Handle"> {
    let mnemonic = "handle";
    let summary = "handle type";
}

#endif
