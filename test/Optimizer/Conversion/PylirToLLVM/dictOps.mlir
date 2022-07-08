// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.str = #py.type

func.func @lookup(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant(#py.str<"key">)
    %1, %2 = py.dict.tryGetItem %arg0[%0]
    return %1 : !py.dynamic
}

// CHECK-LABEL: llvm.func @lookup
// CHECK-SAME: %[[DICT:[[:alnum:]]+]]
// CHECK-NEXT: %[[KEY:.*]] = llvm.mlir.addressof
// CHECK-NEXT: %[[RESULT:.*]] = llvm.call @pylir_dict_lookup(%[[DICT]], %[[KEY]])
// CHECK-NEXT: %[[NULL:.*]] = llvm.mlir.null
// CHECK-NEXT: %[[FOUND:.*]] = llvm.icmp "ne" %[[RESULT]], %[[NULL]]
// CHECK-NEXT: llvm.return %[[RESULT]]

// -----

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.str = #py.type

func.func @insert(%arg0 : !py.dynamic, %value : !py.dynamic) {
    %0 = py.constant(#py.str<"key">)
    py.dict.setItem %arg0[%0] to %value
    return
}

// CHECK-LABEL: llvm.func @insert
// CHECK-SAME: %[[DICT:[[:alnum:]]+]]
// CHECK-SAME: %[[VALUE:[[:alnum:]]+]]
// CHECK-NEXT: %[[KEY:.*]] = llvm.mlir.addressof
// CHECK-NEXT: llvm.call @pylir_dict_insert(%[[DICT]], %[[KEY]], %[[VALUE]])
// CHECK-NEXT: llvm.return
