// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, const, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_int = #py.globalValue<builtins.int, const, initializer = #py.type>
py.external @builtins.int, #builtins_int
#builtins_tuple = #py.globalValue<builtins.tuple, const, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

py.func @foo(%value : !py.dynamic) -> index {
    %0 = int_toIndex %value
    return %0 : index
}

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[VALUE:[[:alnum:]]+]]
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[VALUE]][0, 1]
// CHECK: %[[RES:.*]] = llvm.call @mp_get_i64(%[[GEP]])
// CHECK: llvm.return %[[RES]]
