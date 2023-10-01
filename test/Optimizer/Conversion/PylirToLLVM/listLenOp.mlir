// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, const, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_list = #py.globalValue<builtins.list, const, initializer = #py.type>
py.external @builtins.list, #builtins_list
#builtins_tuple = #py.globalValue<builtins.tuple, const, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

py.func @foo(%list : !py.dynamic) -> index {
    %0 = list_len %list
    return %0 : index
}

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[LIST:[[:alnum:]]+]]
// CHECK-NEXT: %[[SIZE_PTR:.*]] = llvm.getelementptr %[[LIST]][0, 1]
// CHECK-NEXT: %[[LEN:.*]] = llvm.load %[[SIZE_PTR]]
// CHECK-NEXT: llvm.return %[[LEN:.*]]
