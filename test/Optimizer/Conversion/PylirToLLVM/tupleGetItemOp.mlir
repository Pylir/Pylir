// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.func @test(%arg : !py.dynamic, %index : index) -> !py.dynamic {
    %0 = tuple_getItem %arg[%index]
    return %0 : !py.dynamic
}

// CHECK-DAG: #[[DESC:.*]] = #llvm.tbaa_type_desc<id = "Python Tuple Elements"{{.*}}>
// CHECK-DAG: #[[$PYTHON_TUPLE:.*]] = #llvm.tbaa_tag<base_type = #[[DESC]], access_type = #[[DESC]], offset = 0>

// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-SAME: %[[INDEX:[[:alnum:]]+]]
// CHECK-NEXT: %[[TRAILING:.*]] = llvm.getelementptr %[[ARG]][0, 2]
// CHECK-NEXT: %[[GEP:.*]] = llvm.getelementptr %[[TRAILING]][0, %[[INDEX]]]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.load %[[GEP]] {tbaa = [#[[$PYTHON_TUPLE]]]}
// CHECK-NEXT: llvm.return %[[RESULT]]
