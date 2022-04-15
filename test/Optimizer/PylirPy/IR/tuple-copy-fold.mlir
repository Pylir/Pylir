// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type

func @test(%arg0 : !py.dynamic, %arg1 : !py.dynamic, %arg2 : !py.dynamic) -> !py.dynamic {
    %2 = py.tuple.copy %arg0 : %arg1
    %3 = py.tuple.copy %2 : %arg2
    return %3 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG2:[[:alnum:]]+]]
// CHECK-NEXT: %[[COPY:.*]] = py.tuple.copy %[[ARG0]] : %[[ARG2]]
// CHECK-NEXT: return %[[COPY]]

func @test2(%arg0 : !py.dynamic) -> !py.dynamic {
	%0 = py.makeTuple (%arg0)
	%1 = py.constant (@builtins.tuple)
	%2 = py.tuple.copy %0 : %1
	return %2 : !py.dynamic
}

// CHECK-LABEL: @test2
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[TUPLE:.*]] = py.makeTuple (%[[ARG0]])
// CHECK-NEXT: return %[[TUPLE]]
