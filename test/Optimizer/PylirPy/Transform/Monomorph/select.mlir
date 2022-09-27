// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.bool = #py.type

func.func @test(%arg1 : i1, %arg2 : i1) -> i1 {
    %arg0 = arith.constant false
    %0 = py.bool.fromI1 %arg1
    %1 = py.bool.fromI1 %arg2
    %2 = arith.select %arg0, %0, %1 : !py.dynamic
    %3 = py.typeOf %2
    %4 = py.constant(#py.ref<@builtins.bool>)
    %5 = py.is %3, %4
    return %5 : i1
}

func.func @__init__() -> i1 {
	%1 = test.random
	%2 = test.random
	%4 = py.call @test(%1, %2) : (i1, i1) -> i1
	return %2 : i1
}

// CHECK: func.func @[[TEST:([[:alnum:]]|_)+]]
// CHECK: %[[C:.*]] = arith.constant true
// CHECK: return %[[C]]

// CHECK-LABEL: @__init__()
// CHECK: py.call @[[TEST]]
