// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.func @test(%arg0 : !py.dynamic, %arg1 : !py.dynamic, %arg2 : !py.dynamic) -> !py.dynamic {
    %2 = tuple_copy %arg0 : %arg1
    %3 = tuple_copy %2 : %arg2
    return %3 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG2:[[:alnum:]]+]]
// CHECK-NEXT: %[[COPY:.*]] = tuple_copy %[[ARG0]] : %[[ARG2]]
// CHECK-NEXT: return %[[COPY]]

#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>

py.func @test2(%arg0 : !py.dynamic) -> !py.dynamic {
	%0 = makeTuple (%arg0)
	%1 = constant(#builtins_tuple)
	%2 = tuple_copy %0 : %1
	return %2 : !py.dynamic
}

// CHECK-LABEL: @test2
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[TUPLE:.*]] = makeTuple (%[[ARG0]])
// CHECK-NEXT: return %[[TUPLE]]

py.func @test3(%arg0 : !py.dynamic) -> !py.dynamic {
	%0 = list_toTuple %arg0
	%1 = constant(#builtins_tuple)
	%2 = tuple_copy %0 : %1
	return %2 : !py.dynamic
}

// CHECK-LABEL: @test3
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[TUPLE:.*]] = list_toTuple %[[ARG0]]
// CHECK-NEXT: return %[[TUPLE]]
