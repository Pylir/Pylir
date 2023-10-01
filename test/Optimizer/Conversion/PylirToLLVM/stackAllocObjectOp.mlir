// RUN: pylir-opt %s -convert-pylir-to-llvm='target-triple=x86_64-w64-windows-gnu' --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, const, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, const, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

py.func @foo() -> index {
    %0 = constant(#builtins_tuple)
    %1 = pyMem.stackAllocObject tuple %0[0]
    %2 = pyMem.initTuple %1 to ()
    %3 = tuple_len %2
    return %3 : index
}

// CHECK-LABEL: llvm.func @foo
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i{{.*}})
// CHECK-NEXT: %[[MEMORY:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<"PyTuple[0]", (ptr<1>, i64, array<0 x ptr<1>>)> : (i{{.*}}) -> !llvm.ptr<1>
// CHECK-NEXT: %[[TUPLE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK-NEXT: llvm.intr.lifetime.start {{[0-9]+}}, %[[MEMORY]]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant
// CHECK-NEXT: "llvm.intr.memset"(%[[MEMORY]], %[[ZERO]], %[[SIZE]])
// CHECK-SAME: isVolatile = false
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 0]
// CHECK-NEXT: llvm.store %[[TUPLE]], %[[GEP]]
