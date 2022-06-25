// RUN: pylir-opt %s --pylir-monomorph --canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @aType = #py.type

func @createObject(%typeObject : !py.dynamic) -> !py.dynamic {
	%0 = py.makeObject %typeObject
	return %0 : !py.dynamic
}

func @test() -> !py.dynamic {
	%0 = py.constant(@aType)
	%1 = py.call @createObject(%0) : (!py.dynamic) -> !py.dynamic
	%2 = py.typeOf %1
	return %2 : !py.dynamic
}

// CHECK-LABEL: func @test
// CHECK-DAG: %[[INT:.*]] = py.constant(@aType)
// CHECK-DAG: return %[[INT]]
