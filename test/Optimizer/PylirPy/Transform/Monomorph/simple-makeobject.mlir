// RUN: pylir-opt %s --pylir-monomorph --canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @aType = #py.type

py.func @createObject(%typeObject : !py.dynamic) -> !py.dynamic {
	%0 = makeObject %typeObject
	test.use(%0) : !py.dynamic // a kind of side effect
	return %0 : !py.dynamic
}

py.func @test() -> !py.dynamic {
	%0 = constant(#py.ref<@aType>)
	%1 = call @createObject(%0) : (!py.dynamic) -> !py.dynamic
	%2 = typeOf %1
	return %2 : !py.dynamic
}

// CHECK-LABEL: func @test
// CHECK-NEXT: %[[INT:.*]] = constant(#py.ref<@aType>)
// CHECK-NEXT: call @[[SPECIALIZATION:[[:alnum:]]+]]
// CHECK: return %[[INT]]

// CHECK: func private @[[SPECIALIZATION]]
// CHECK-NEXT: %[[C:.*]] = constant(#py.ref<@aType>)
// CHECK-NEXT: %[[O:.*]] = makeObject %[[C]]
// CHECK-NEXT: test.use(%[[O]])
// CHECK-NEXT: return %[[O]]
