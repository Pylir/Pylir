// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.tuple = #py.type

func.func @foo(%value : !py.dynamic, %arg0 : !py.dynamic) -> !py.dynamic {
    %2 = py.function.call %value(%arg0)
    return %2 : !py.dynamic
}

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[VALUE:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[VALUE]][0, 1]
// CHECK-NEXT: %[[PTR:.*]] = llvm.load %[[GEP]] {tbaa = [@tbaa::@"Python Function Pointer access"]}
// CHECK-NEXT: %[[RES:.*]] = llvm.call %[[PTR]](%[[ARG0]])
// CHECK-NEXT: llvm.return %[[RES]]