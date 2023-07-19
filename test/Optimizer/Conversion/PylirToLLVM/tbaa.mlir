// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.func @test(%arg : !py.dynamic) -> index {
    %0 = tuple_len %arg
    return %0 : index
}

// CHECK-DAG: #[[ROOT:.*]] = #llvm.tbaa_root<id = "Pylir TBAA Root">
// CHECK-DAG: #[[DESC:.*]] = #llvm.tbaa_type_desc<id = "Python Tuple Size", members = {<#[[ROOT]], 0>}>
// CHECK-DAG: #[[$TAG:.*]] = #llvm.tbaa_tag<base_type = #[[DESC]], access_type = #[[DESC]], offset = 0>

// CHECK-LABEL: @test
// CHECK: llvm.load %{{.*}} {tbaa = [#[[$TAG]]]}
