// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.func @test(%arg : !py.dynamic) -> !py.dynamic {
    %0 = type_slots %arg
    return %0 : !py.dynamic
}

// CHECK-DAG: #[[DESC:.*]] = #llvm.tbaa_type_desc<id = "Python Type Instance Slots"{{.*}}>
// CHECK-DAG: #[[$PYTHON_TYPE_INSTANCE_SLOTS:.*]] = #llvm.tbaa_tag<base_type = #[[DESC]], access_type = #[[DESC]], offset = 0>

// CHECK: @test
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[MRO:.*]] = llvm.getelementptr %[[ARG]][0, 4]
// CHECK-NEXT: %[[RESULT:.*]] = llvm.load %[[MRO]] {tbaa = [#[[$PYTHON_TYPE_INSTANCE_SLOTS]]]}
// CHECK-NEXT: llvm.return %[[RESULT]]
