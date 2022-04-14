
#ifndef PYLIR_PY_BASE_TABLEGEN
#define PYLIR_PY_BASE_TABLEGEN

include "mlir/IR/OpBase.td"

def PylirPy_Dialect : Dialect {
    let name = "py";
    let summary = "A dialect representing python programs";
    let description = [{
        TODO
    }];
    let cppNamespace = "::pylir::Py";

    let dependentDialects = ["::mlir::arith::ArithmeticDialect", "::mlir::cf::ControlFlowDialect"];

    let hasConstantMaterializer = 1;
    let hasCanonicalizer = 1;

    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;

    let extraClassDeclaration = [{
        void initializeAttributes();

        void initializeTypes();
    }];

    let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

#endif
