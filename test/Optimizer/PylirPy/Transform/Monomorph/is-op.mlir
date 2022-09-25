// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

func.func @test(%arg0 : !py.dynamic) -> i1 {
    %0 = py.constant(#py.ref<@builtins.type>)
    %1 = py.is %arg0, %0
    return %1 : i1
}

func.func @__init__() -> i1 {
	%1 = py.constant(#py.ref<@builtins.str>)
	%2 = py.call @test(%1) : (!py.dynamic) -> i1
	return %2 : i1
}

// CHECK-LABEL: @__init__()
// CHECK: py.call @[[TEST:([[:alnum:]]|_)+]](

// CHECK: func.func private @[[TEST]]
// CHECK: %[[C:.*]] = arith.constant false
// CHECK: return %[[C]]
