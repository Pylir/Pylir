// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.func @test(%arg0 : !py.dynamic, %arg1 : !py.dynamic, %arg2 : !py.dynamic, %arg3 : !py.dynamic) -> !py.dynamic {
    %0 = str_copy %arg1 : %arg0
    %1 = str_copy %arg2 : %arg0
    %2 = str_copy %arg3 : %arg0
    %4 = str_concat %0, %1, %2
    return %4 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG2:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG3:[[:alnum:]]+]]
// CHECK-NEXT: %[[RES:.*]] = str_concat %[[ARG1]], %[[ARG2]], %[[ARG3]]
// CHECK-NEXT: return %[[RES]]

// -----

py.func @test(%arg0 : !py.dynamic, %arg1 : !py.dynamic, %arg2 : !py.dynamic) -> !py.dynamic {
    %0 = str_copy %arg1 : %arg0
    %1 = str_copy %0 : %arg2
    return %1 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG2:[[:alnum:]]+]]
// CHECK-NEXT: %[[RES:.*]] = str_copy %[[ARG1]] : %[[ARG2]]
// CHECK-NEXT: return %[[RES]]

// -----

py.func @test(%arg0 : !py.dynamic, %arg1 : !py.dynamic, %arg2 : !py.dynamic) -> i1 {
    %0 = str_copy %arg1 : %arg0
    %1 = str_copy %arg2 : %arg0
    %2 = str_equal %0, %1
    return %2 : i1
}

// CHECK-LABEL: @test
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG2:[[:alnum:]]+]]
// CHECK-NEXT: %[[RES:.*]] = str_equal %[[ARG1]], %[[ARG2]]
// CHECK-NEXT: return %[[RES]]

// -----

py.func @test(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> index {
    %0 = str_copy %arg1 : %arg0
    %1 = str_hash %0
    return %1 : index
}

// CHECK-LABEL: @test
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: %[[RES:.*]] = str_hash %[[ARG1]]
// CHECK-NEXT: return %[[RES]]
