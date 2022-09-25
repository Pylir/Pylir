// RUN: pylir-opt %s --pylir-monomorph --canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @aType = #py.type

func.func @createObject(%typeObject : !py.dynamic) -> !py.dynamic {
	%0 = py.makeObject %typeObject
	test.use(%0) : !py.dynamic // a kind of side effect
	return %0 : !py.dynamic
}

func.func @test() -> !py.dynamic {
	%0 = py.constant(#py.ref<@aType>)
	%1 = py.call @createObject(%0) : (!py.dynamic) -> !py.dynamic
	%2 = py.typeOf %1
	return %2 : !py.dynamic
}

// CHECK-LABEL: func @test
// CHECK-NEXT: %[[INT:.*]] = py.constant(#py.ref<@aType>)
// CHECK-NEXT: py.call @[[SPECIALIZATION:[[:alnum:]]+]]
// CHECK: return %[[INT]]

// CHECK: func private @[[SPECIALIZATION]]
// CHECK-NEXT: %[[C:.*]] = py.constant(#py.ref<@aType>)
// CHECK-NEXT: %[[O:.*]] = py.makeObject %[[C]]
// CHECK-NEXT: test.use(%[[O]])
// CHECK-NEXT: return %[[O]]
