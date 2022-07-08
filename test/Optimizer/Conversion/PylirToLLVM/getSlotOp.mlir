// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type<slots = {__slots__ = #py.tuple<(#py.str<"__slots__">)>}>
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.object = #py.type
py.globalValue @builtins.tuple = #py.type

func.func @foo() -> !py.dynamic {
    %0 = py.constant(@builtins.type)
    %1 = py.constant(@builtins.tuple)
    %2 = py.getSlot "__slots__" from %1 : %0
    return %2 : !py.dynamic
}

// CHECK-LABEL: @foo
// CHECK-NEXT: %[[TYPE:.*]] = llvm.mlir.addressof @builtins.type
// CHECK-NEXT: %[[TUPLE:.*]] = llvm.mlir.addressof @builtins.tuple
// CHECK-NEXT: %[[SIZE:.*]] = llvm.mlir.constant({{[0-9]+}} : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TUPLE]][%[[SIZE]]] : {{.*}}, i8
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : i{{[0-9]+}})
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP]][%[[ZERO]]]
// CHECK-NEXT: %[[LOAD:.*]] = llvm.load %[[GEP2]]
// CHECK-NEXT: llvm.return %[[LOAD]]

// -----

py.globalValue @builtins.type = #py.type<slots = {__slots__ = #py.tuple<(#py.str<"__slots__">)>}>
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

func.func @foo(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> !py.dynamic {
    %0 = py.getSlot "__slots__" from %arg0 : %arg1
    return %0 : !py.dynamic
}

// CHECK-LABEL: llvm.func @foo
