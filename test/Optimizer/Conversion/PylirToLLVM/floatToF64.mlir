// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.float = #py.type
py.globalValue const @builtins.tuple = #py.type

func.func @foo(%value : !py.dynamic) -> f64 {
    %2 = py.float.toF64 %value
    return %2 : f64
}

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[VALUE:[[:alnum:]]+]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[VALUE]][0, 1]
// CHECK-NEXT: %[[RES:.*]] = llvm.load %[[GEP]]
// CHECK-NEXT: llvm.return %[[RES]]
