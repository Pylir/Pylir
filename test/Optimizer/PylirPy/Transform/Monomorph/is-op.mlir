// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.str = #py.type

py.func @test(%arg0 : !py.dynamic) -> i1 {
    %0 = constant(#py.ref<@builtins.type>)
    %1 = is %arg0, %0
    return %1 : i1
}

py.func @__init__() -> i1 {
	%1 = constant(#py.ref<@builtins.str>)
	%2 = call @test(%1) : (!py.dynamic) -> i1
	return %2 : i1
}

// CHECK-LABEL: @__init__()
// CHECK: call @[[TEST:([[:alnum:]]|_)+]](

// CHECK: py.func private @[[TEST]]
// CHECK: %[[C:.*]] = arith.constant false
// CHECK: return %[[C]]
