// RUN: pylir-tblgen -gen-op-variable-decorators %s | FileCheck %s

include "mlir/IR/OpBase.td"
include "pylir/Optimizer/Util/OpVariableBase.td"

def Test_Dialect : Dialect {
  let name = "test";
}

class Test_Op<string mnemonic, list<Trait> traits = []>
  : Op<Test_Dialect, mnemonic, traits>;

def TestInterface : OpInterface<"TestInterface"> {
  let methods = [
    ExtendableInterfaceMethod<"", "int", "lol", (ins), [{
      int i = 0;
    }], [{
      return i;
    }]>
  ];
}

def TestVariableGen : OpVariableGen<TestInterface> {
  let methodGens = [
    OpVariableMethodGen<"lol", [{
      i += $_arg.getOperandNumber();
    }]>
  ];
}

def Test_A : Test_Op<"a", [DeclareOpInterfaceMethods<TestInterface>]> {
  let arguments = (ins I32Attr:$test, Arg<I32, "", [TestVariableGen]>:$test2);
}

def Test_B : Test_Op<"b", [TestInterface]> {
  let arguments = (ins I32Attr:$test);
}

def Test_C : Test_Op<"c", [DeclareOpInterfaceMethods<TestInterface>]> {
  let arguments = (ins I32Attr:$test);
}

// CHECK-LABEL: int A::lol()
// CHECK-NEXT: int i = 0;
// CHECK: for (::mlir::OpOperand& tblGenOdsIter : llvm::MutableArrayRef<::mlir::OpOperand>(getTest2Mutable())) {
// CHECK-NEXT:     i += tblGenOdsIter.getOperandNumber();
// CHECK-NEXT: }
// CHECK: return i;

// CHECK-NOT: B::lol

// CHECK-LABEL: int C::lol()
// CHECK-NEXT: int i = 0;
// CHECK-NEXT: return i;

def Test_D : Test_Op<"d", [DeclareOpInterfaceMethods<TestInterface>]> {
  let arguments = (ins
    Arg<Variadic<I32>, "", [TestVariableGen]>:$test1,
    Arg<Optional<I32>, "", [TestVariableGen]>:$test2
  );
}

// CHECK-LABEL: int D::lol()
// CHECK-NEXT: int i = 0;
// CHECK: for (::mlir::OpOperand& tblGenOdsIter : llvm::MutableArrayRef<::mlir::OpOperand>(getTest1Mutable())) {
// CHECK-NEXT:     i += tblGenOdsIter.getOperandNumber();
// CHECK-NEXT: }
// CHECK: for (::mlir::OpOperand& tblGenOdsIter : llvm::MutableArrayRef<::mlir::OpOperand>(getTest2Mutable())) {
// CHECK-NEXT:     i += tblGenOdsIter.getOperandNumber();
// CHECK-NEXT: }
// CHECK: return i;

def Test_E : Test_Op<"e", [DeclareOpInterfaceMethods<TestInterface, ["lol"]>]> {
  let arguments = (ins I32Attr:$test);
}

// CHECK-NOT: E::lol
