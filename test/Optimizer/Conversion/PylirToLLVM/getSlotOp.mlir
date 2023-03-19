// RUN: pylir-opt %s  -convert-arith-to-llvm -convert-pylir-to-llvm --reconcile-unrealized-casts --split-input-file --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type<instance_slots = #py.tuple<(#py.str<"__eq__">)>>
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.object = #py.type
py.globalValue @builtins.tuple = #py.type

py.func @foo() -> !py.dynamic {
    %0 = constant(#py.ref<@builtins.tuple>)
    %c0 = arith.constant 0 : index
    %2 = getSlot %0[%c0]
    return %2 : !py.dynamic
}

// CHECK-LABEL: @foo
// CHECK-NEXT: %[[TUPLE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : {{.*}}) : i{{[0-9]+}}
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE]][0, 0]
// CHECK-NEXT: %[[TYPE:.*]] = llvm.load %[[GEP]] {tbaa = [@tbaa::@"Python Type Object access"]}
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TYPE]][0, 1]
// CHECK-NEXT: %[[OFFSET:.*]] = llvm.load %[[GEP]] {tbaa = [@tbaa::@"Python Type Offset access"]}
// CHECK-NEXT: %[[ADD:.*]] = llvm.add %[[OFFSET]], %[[ZERO]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE]][%[[ADD]]]
// CHECK-NEXT: %[[LOAD:.*]] = llvm.load %[[GEP]] {tbaa = [@tbaa::@"Python Object Slots access"]}
// CHECK-NEXT: llvm.return %[[LOAD]]
