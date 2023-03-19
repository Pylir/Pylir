// RUN: pylir-opt %s --pylir-monomorph --canonicalize --split-input-file | FileCheck %s

py.globalValue const @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue const @builtins.int = #py.type

py.func @plusOne(%arg0 : !py.dynamic) -> !py.dynamic {
	%0 = call @header(%arg0) : (!py.dynamic) -> !py.dynamic
	return %0 : !py.dynamic
}

py.func @plusTwo(%arg0 : !py.dynamic) -> !py.dynamic {
	%0 = call @plusOne(%arg0) : (!py.dynamic) -> !py.dynamic
    return %0 : !py.dynamic
}

py.func @header(%arg0 : !py.dynamic) -> !py.dynamic {
	%0 = test.random
	cf.cond_br %0, ^bb1, ^bb2

^bb1:
	%1 = call @plusOne(%arg0) : (!py.dynamic) -> !py.dynamic
	return %1 : !py.dynamic

^bb2:
	%2 = call @plusTwo(%arg0) : (!py.dynamic) -> !py.dynamic
    return %2 : !py.dynamic
}

py.func @root() -> !py.dynamic {
	%0 = constant(#py.int<1>)
	%1 = call @header(%0) : (!py.dynamic) -> !py.dynamic
	%2 = typeOf %1
	return %2 : !py.dynamic
}

// CHECK-LABEL: func @root

// TODO: probably some unbound value if we want to propagate it that way
