
#ifndef PYLIR_TEST_TABLEGEN
#define PYLIR_TEST_TABLEGEN

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TestDialect : Dialect {
    let name = "test";
    let cppNamespace = "::pylir::test";
}

class Test_Op<string mnemonic, list <Trait> traits = []> : Op<TestDialect, mnemonic, traits>;

def RandomOp : Test_Op<"random", [NoSideEffect]> {
    let arguments = (ins);
    let results = (outs I1:$result);

    let assemblyFormat = "attr-dict";
}

#endif
