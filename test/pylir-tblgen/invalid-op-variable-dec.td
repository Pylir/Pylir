// RUN: not pylir-tblgen -gen-op-variable-decorators %s 2>&1 | FileCheck %s

include "mlir/IR/OpBase.td"
include "pylir/Optimizer/Util/OpVariableBase.td"

def Test_Dialect : Dialect {
  let name = "test";
}

class Test_Op<string mnemonic, list<Trait> traits = []>
  : Op<Test_Dialect, mnemonic, traits>;

def TestInterface : Interface<"TestInterface"> {
  let methods = [
    InterfaceMethod<"", "void", "thing">
  ];
}

def TestVariableGen : OpVariableGen<TestInterface> {
  let methodGens = [
    OpVariableMethodGen<"lol", "">,
    OpVariableMethodGen<"thing", "">,
  ];
}

def Test_A : Test_Op<"a"> {
  let arguments = (ins
    Arg<I32Attr, "", [TestVariableGen]>:$test,
    Arg<I32, "", [TestVariableGen]>:$test2
  );
}

// CHECK: error: 'OpVariableGen' is currently only supported on operands
// CHECK: No method named 'lol' found in interface 'TestInterface'
// CHECK: 'OpVariableGen' can only extend interface method of type 'ExtendableInterfaceMethod'
