// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.int = #py.type
py.globalValue const @builtins.tuple = #py.type

py.func @foo(%value : !py.dynamic) -> index {
    %0 = int_toIndex %value
    return %0 : index
}

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[VALUE:[[:alnum:]]+]]
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[VALUE]][0, 1]
// CHECK: %[[RES:.*]] = llvm.call @mp_get_i64(%[[GEP]])
// CHECK: llvm.return %[[RES]]
