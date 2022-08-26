// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.list = #py.type
py.globalValue const @builtins.tuple = #py.type

func.func @foo(%list : !py.dynamic) -> index {
    %0 = py.list.len %list
    return %0 : index
}

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[LIST:[[:alnum:]]+]]
// CHECK-NEXT: %[[SIZE_PTR:.*]] = llvm.getelementptr %[[LIST]][0, 1]
// CHECK-NEXT: %[[LEN:.*]] = llvm.load %[[SIZE_PTR]]
// CHECK-NEXT: llvm.return %[[LEN:.*]]
