// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.func @foo(%value : !py.dynamic) -> f64 {
    %2 = float_toF64 %value
    return %2 : f64
}

// CHECK-DAG: #[[DESC:.*]] = #llvm.tbaa_type_desc<id = "Python Float Value"{{.*}}>
// CHECK-DAG: #[[$PYTHON_FLOAT_VALUE:.*]] = #llvm.tbaa_tag<base_type = #[[DESC]], access_type = #[[DESC]], offset = 0>

// CHECK-LABEL: llvm.func @foo
// CHECK-SAME: %[[VALUE:[[:alnum:]]+]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[VALUE]][0, 1]
// CHECK-NEXT: %[[RES:.*]] = llvm.load %[[GEP]] {tbaa = [#[[$PYTHON_FLOAT_VALUE]]]}
// CHECK-NEXT: llvm.return %[[RES]]
