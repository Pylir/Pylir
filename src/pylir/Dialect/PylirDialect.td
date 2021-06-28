
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

class Pylir_Type<string name,string baseCppClass = "::mlir::Type"> :
    TypeDef<Pylir_Dialect, name,baseCppClass>;

def Pylir_Integer : Pylir_Type<"PylirInteger"> {
    let mnemonic = "pylir.int";
}

def Pylir_None : Pylir_Type<"PylirNone"> {
    let mnemonic = "pylir.none";
}

def Pylir_String : Pylir_Type<"PylirString"> {
    let mnemonic = "pylir.str";
}

def Pylir_List : Pylir_Type<"PylirList"> {
    let mnemonic = "pylir.list";
}

def Pylir_Set : Pylir_Type<"PylirSet"> {
    let mnemonic = "pylir.set";
}

def Pylir_Tuple : Pylir_Type<"PylirTuple"> {
    let mnemonic = "pylir.tuple";
}

def Pylir_Dict : Pylir_Type<"PylirDict"> {
    let mnemonic = "pylir.dict";
}

def Pylir_Variant : Pylir_Type<"PylirVariant"> {
    let mnemonic = "pylir.variant";
    let parameters = (ins ArrayRefParameter<"mlir::Type", "alternatives">
                          : $types);
}

#endif
