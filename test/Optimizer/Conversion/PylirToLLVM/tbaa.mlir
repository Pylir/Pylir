// RUN: pylir-opt %s -convert-pylir-to-llvm --split-input-file | FileCheck %s

py.func @foo() {
    return
}

// CHECK-LABEL: llvm.func @foo

// CHECK-NOT: llvm.metadata @tbaa

// -----

py.func @test(%arg : !py.dynamic) -> index {
    %0 = tuple_len %arg
    return %0 : index
}

// CHECK: @test
// CHECK: llvm.load %{{.*}} {tbaa = [@tbaa::@"Python Tuple Size access"]}

// CHECK-LABEL: llvm.metadata @tbaa
// CHECK-DAG: llvm.tbaa_root @root {id = "Pylir TBAA Root"}
// CHECK-DAG: llvm.tbaa_type_desc @"Python Tuple Size type" {id = "Python Tuple Size type", members = {<@root, 0>}}
// CHECK-DAG: llvm.tbaa_tag @"Python Tuple Size access" {access_type = @"Python Tuple Size type", base_type = @"Python Tuple Size type", offset = 0 : i64}
