// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.func @test(%arg : !py.dynamic) -> index {
    %0 = tuple_len %arg
    return %0 : index
}

// CHECK-DAG: #[[DESC:.*]] = #llvm.tbaa_type_desc<id = "Python Tuple Size"{{.*}}>
// CHECK-DAG: #[[$PYTHON_TUPLE_SIZE:.*]] = #llvm.tbaa_tag<base_type = #[[DESC]], access_type = #[[DESC]], offset = 0>

// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[ARG]][0, 1]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.load %[[GEP]] {tbaa = [#[[$PYTHON_TUPLE_SIZE]]]}
// CHECK-NEXT: llvm.return %[[RESULT]]
