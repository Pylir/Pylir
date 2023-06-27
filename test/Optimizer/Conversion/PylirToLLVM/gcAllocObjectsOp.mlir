// RUN: pylir-opt %s -convert-arith-to-llvm -convert-pylir-to-llvm --reconcile-unrealized-casts --split-input-file | FileCheck %s


py.globalValue const @builtins.type = #py.type<slots = {__slots__ = #py.tuple<(#py.str<"__slots__">,#py.str<"__eq__">,#py.str<"__hash__">)>}>

py.globalValue const @builtins.tuple = #py.type // stub
py.globalValue const @builtins.str = #py.type // stub

py.func @foo() -> !pyMem.memory {
    %0 = constant(#py.ref<@builtins.str>)
    %c0 = arith.constant 0 : index
    %1 = pyMem.gcAllocObject %0[%c0]
    return %1 : !pyMem.memory
}

// CHECK-LABEL: llvm.func @foo
// CHECK-NEXT: %[[STR:.*]] = llvm.mlir.addressof @builtins.str
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : index)
// CHECK-NEXT: %[[INSTANCE_SIZE:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[POINTER_SIZE:.*]] = llvm.mlir.constant
// CHECK-NEXT: %[[MUL:.*]] = llvm.mul %[[ZERO]], %[[POINTER_SIZE]]
// CHECK-NEXT: %[[BYTES:.*]] = llvm.add %[[MUL]], %[[INSTANCE_SIZE]]
// CHECK-NEXT: %[[MEMORY:.*]] = llvm.call @pylir_gc_alloc(%[[BYTES]])
// CHECK-NEXT: %[[ZERO_I8:.*]] = llvm.mlir.constant(0 : i8)
// CHECK-NEXT: "llvm.intr.memset"(%[[MEMORY]], %[[ZERO_I8]], %[[BYTES]])
// CHECK-SAME: isVolatile = false
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[MEMORY]][0, 0]
// CHECK-NEXT: llvm.store %[[STR]], %[[GEP]] {tbaa = [@tbaa::@"Python Type Object access"]}
// CHECK-NEXT: llvm.return %[[MEMORY]]

// CHECK: llvm.func @pylir_gc_alloc(i{{[0-9]+}}) -> (!llvm.ptr<{{[0-9]+}}> {
// CHECK-SAME: llvm.noalias
// CHECK-SAME: }) attributes

// -----

py.globalValue const @builtins.type = #py.type<slots = {__slots__ = #py.tuple<(#py.str<"__slots__">,#py.str<"__eq__">,#py.str<"__hash__">)>}>

py.globalValue const @builtins.tuple = #py.type // stub
py.globalValue const @builtins.str = #py.type // stub

py.func @foo(%arg0 : !py.dynamic) -> !pyMem.memory {
    %c0 = arith.constant 0 : index
    %0 = pyMem.gcAllocObject %arg0[%c0]
    return %0 : !pyMem.memory
}

// CHECK-LABEL: llvm.func @foo
