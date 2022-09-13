// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.str = #py.type

func.func @lookup(%arg0 : !py.dynamic, %hash : index) -> !py.dynamic {
    %0 = py.constant(#py.str<"key">)
    %1 = py.dict.tryGetItem %arg0[%0 hash(%hash)]
    return %1 : !py.dynamic
}

// CHECK-LABEL: llvm.func @lookup
// CHECK-SAME: %[[DICT:[[:alnum:]]+]]
// CHECK-SAME: %[[HASH:[[:alnum:]]+]]
// CHECK-NEXT: %[[KEY:.*]] = llvm.mlir.addressof
// CHECK-NEXT: %[[RESULT:.*]] = llvm.call @pylir_dict_lookup(%[[DICT]], %[[KEY]], %[[HASH]])
// CHECK-NEXT: llvm.return %[[RESULT]]

// -----

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.str = #py.type

func.func @insert(%arg0 : !py.dynamic, %value : !py.dynamic, %hash : index) {
    %0 = py.constant(#py.str<"key">)
    py.dict.setItem %arg0[%0 hash(%hash)] to %value
    return
}

// CHECK-LABEL: llvm.func @insert
// CHECK-SAME: %[[DICT:[[:alnum:]]+]]
// CHECK-SAME: %[[VALUE:[[:alnum:]]+]]
// CHECK-SAME: %[[HASH:[[:alnum:]]+]]
// CHECK-NEXT: %[[KEY:.*]] = llvm.mlir.addressof
// CHECK-NEXT: llvm.call @pylir_dict_insert(%[[DICT]], %[[KEY]], %[[HASH]], %[[VALUE]])
// CHECK-NEXT: llvm.return
