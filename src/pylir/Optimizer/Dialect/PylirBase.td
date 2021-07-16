
#ifndef PYLIR_BASE_TABLEGEN
#define PYLIR_BASE_TABLEGEN

include "mlir/IR/OpBase.td"

def Pylir_Dialect : Dialect {
    let name = "pylir";
    let summary = "A dialect representing python programs";
    let description = [{
        TODO
    }];
    let cppNamespace = "::pylir::Dialect";

    let hasConstantMaterializer = 1;

    let extraClassDeclaration = [{
    mlir::Attribute parseAttribute(mlir::DialectAsmParser& parser, mlir::Type type) const override;

    void printAttribute(mlir::Attribute attriubte, mlir::DialectAsmPrinter& printer) const override;
    }];
}

#endif
