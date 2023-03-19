// RUN: pylir-opt %s --pylir-monomorph --canonicalize --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue const @builtins.int = #py.type
py.globalValue const @builtins.tuple = #py.type

py.func @other(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = call @start(%arg0) : (!py.dynamic) -> !py.dynamic
    return %0 : !py.dynamic
}

py.func @start(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = makeTuple (%arg0)
    %1 = call @other(%0)  : (!py.dynamic) -> !py.dynamic
    return %1 : !py.dynamic
}

py.func @root() -> !py.dynamic {
	%0 = constant(#py.tuple<(#py.int<0>)>)
	%1 = call @start(%0) : (!py.dynamic) -> !py.dynamic
	%2 = typeOf %1
	return %2 : !py.dynamic
}

// CHECK-LABEL: @root
// CHECK-NEXT: %[[C:.*]] = constant
// CHECK-NEXT: %[[RES:.*]] = call @start(%[[C]])
// CHECK-NEXT: %[[TYPE:.*]] = typeOf %[[RES]]
// CHECK-NEXT: return %[[TYPE]]
