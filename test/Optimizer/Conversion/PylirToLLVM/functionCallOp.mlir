// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.func @foo(%value : !py.dynamic, %arg0 : !py.dynamic) -> !py.dynamic {
    %2 = function_call %value(%arg0)
    return %2 : !py.dynamic
}

// CHECK-DAG: #[[DESC:.*]] = #llvm.tbaa_type_desc<id = "Python Function Pointer"{{.*}}>
// CHECK-DAG: #[[$PYTHON_FUNCTION:.*]] = #llvm.tbaa_tag<base_type = #[[DESC]], access_type = #[[DESC]], offset = 0>

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[VALUE:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[VALUE]][0, 1]
// CHECK-NEXT: %[[PTR:.*]] = llvm.load %[[GEP]] {tbaa = [#[[$PYTHON_FUNCTION]]]}
// CHECK-NEXT: %[[RES:.*]] = llvm.call %[[PTR]](%[[ARG0]])
// CHECK-NEXT: llvm.return %[[RES]]
