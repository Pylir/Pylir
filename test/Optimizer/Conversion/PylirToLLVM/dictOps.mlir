// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

py.func @lookup(%arg0 : !py.dynamic, %hash : index) -> !py.dynamic {
    %0 = constant(#py.str<"key">)
    %1 = dict_tryGetItem %arg0[%0 hash(%hash)]
    return %1 : !py.dynamic
}

// CHECK-LABEL: llvm.func @lookup
// CHECK-SAME: %[[DICT:[[:alnum:]]+]]
// CHECK-SAME: %[[HASH:[[:alnum:]]+]]
// CHECK-NEXT: %[[KEY:.*]] = llvm.mlir.addressof
// CHECK-NEXT: %[[RESULT:.*]] = llvm.call @pylir_dict_lookup(%[[DICT]], %[[KEY]], %[[HASH]])
// CHECK-NEXT: llvm.return %[[RESULT]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

py.func @insert(%arg0 : !py.dynamic, %value : !py.dynamic, %hash : index) {
    %0 = constant(#py.str<"key">)
    dict_setItem %arg0[%0 hash(%hash)] to %value
    return
}

// CHECK-LABEL: llvm.func @insert
// CHECK-SAME: %[[DICT:[[:alnum:]]+]]
// CHECK-SAME: %[[VALUE:[[:alnum:]]+]]
// CHECK-SAME: %[[HASH:[[:alnum:]]+]]
// CHECK-NEXT: %[[KEY:.*]] = llvm.mlir.addressof
// CHECK-NEXT: llvm.call @pylir_dict_insert(%[[DICT]], %[[KEY]], %[[HASH]], %[[VALUE]])
// CHECK-NEXT: llvm.return
