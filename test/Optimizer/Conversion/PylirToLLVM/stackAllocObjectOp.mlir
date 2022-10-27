// RUN: pylir-opt %s -convert-pylir-to-llvm='target-triple=x86_64-w64-windows-gnu' --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.tuple = #py.type

func.func @foo() -> index {
    %0 = py.constant(#py.ref<@builtins.tuple>)
    %1 = pyMem.stackAllocObject tuple %0[0]
    %2 = pyMem.initTuple %1 to ()
    %3 = py.tuple.len %2
    return %3 : index
}

// CHECK-LABEL: llvm.func @foo
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i{{.*}})
// CHECK-NEXT: %[[MEMORY:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(ptr<1>, i64, array<0 x ptr<1>>)> : (i{{.*}}) -> !llvm.ptr<1>
// CHECK-NEXT: %[[TUPLE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK-NEXT: llvm.intr.lifetime.start {{[0-9]+}}, %[[MEMORY]]
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[FALSE:.*]] = llvm.mlir.constant
// CHECK-NEXT: "llvm.intr.memset"(%[[MEMORY]], %[[ZERO]], %[[SIZE]], %[[FALSE]])
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 0]
// CHECK-NEXT: llvm.store %[[TUPLE]], %[[GEP]]