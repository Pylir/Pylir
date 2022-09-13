// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

func.func @test(%arg0 : !py.dynamic, %arg1 : !py.dynamic, %arg2 : !py.dynamic, %arg3 : !py.dynamic) -> !py.dynamic {
    %0 = py.str.copy %arg1 : %arg0
    %1 = py.str.copy %arg2 : %arg0
    %2 = py.str.copy %arg3 : %arg0
    %4 = py.str.concat %0, %1, %2
    return %4 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG2:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG3:[[:alnum:]]+]]
// CHECK-NEXT: %[[RES:.*]] = py.str.concat %[[ARG1]], %[[ARG2]], %[[ARG3]]
// CHECK-NEXT: return %[[RES]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

func.func @test(%arg0 : !py.dynamic, %arg1 : !py.dynamic, %arg2 : !py.dynamic) -> !py.dynamic {
    %0 = py.str.copy %arg1 : %arg0
    %1 = py.str.copy %0 : %arg2
    return %1 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG2:[[:alnum:]]+]]
// CHECK-NEXT: %[[RES:.*]] = py.str.copy %[[ARG1]] : %[[ARG2]]
// CHECK-NEXT: return %[[RES]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

func.func @test(%arg0 : !py.dynamic, %arg1 : !py.dynamic, %arg2 : !py.dynamic) -> i1 {
    %0 = py.str.copy %arg1 : %arg0
    %1 = py.str.copy %arg2 : %arg0
    %2 = py.str.equal %0, %1
    return %2 : i1
}

// CHECK-LABEL: @test
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG2:[[:alnum:]]+]]
// CHECK-NEXT: %[[RES:.*]] = py.str.equal %[[ARG1]], %[[ARG2]]
// CHECK-NEXT: return %[[RES]]

// -----


py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

func.func @test(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> index {
    %0 = py.str.copy %arg1 : %arg0
    %1 = py.str.hash %0
    return %1 : index
}

// CHECK-LABEL: @test
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-NEXT: %[[RES:.*]] = py.str.hash %[[ARG1]]
// CHECK-NEXT: return %[[RES]]
