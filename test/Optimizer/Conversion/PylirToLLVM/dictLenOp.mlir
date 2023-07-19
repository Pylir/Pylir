// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.func @test(%arg : !py.dynamic) -> index {
    %0 = dict_len %arg
    return %0 : index
}

// CHECK-DAG: #[[DESC:.*]] = #llvm.tbaa_type_desc<id = "Python Dict Size"{{.*}}>
// CHECK-DAG: #[[$PYTHON_DICT_SIZE:.*]] = #llvm.tbaa_tag<base_type = #[[DESC]], access_type = #[[DESC]], offset = 0>

// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ARG]][0, 1]
// CHECK-NEXT: %[[GEP2:.*]] = llvm.getelementptr %[[GEP1]][0, 0]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.load %[[GEP2]] {tbaa = [#[[$PYTHON_DICT_SIZE]]]}
// CHECK-NEXT: llvm.return %[[RESULT]]
