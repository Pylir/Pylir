// RUN: pylir-opt %s --pylir-monomorph --canonicalize --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.int = #py.type
py.globalValue const @builtins.tuple = #py.type

func.func private @header(%arg0 : !py.dynamic) -> !py.dynamic

func.func @root() -> !py.dynamic {
	%0 = py.constant(#py.tuple<(#py.int<0>)>)
	%1 = py.call @header(%0) : (!py.dynamic) -> !py.dynamic
	%2 = py.typeOf %1
	return %2 : !py.dynamic
}

// CHECK-LABEL: @root
// CHECK-NEXT: %[[C:.*]] = py.constant
// CHECK-NEXT: %[[RES:.*]] = py.call @header(%[[C]])
// CHECK-NEXT: %[[TYPE:.*]] = py.typeOf %[[RES]]
// CHECK-NEXT: return %[[TYPE]]
