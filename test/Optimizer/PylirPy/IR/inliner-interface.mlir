// RUN: pylir-opt %s --test-inliner-interface | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.BaseException = #py.type

func private @create_exception() -> !py.dynamic

func @inline_foo(%arg0 : i1) -> !py.dynamic {
	%0 = py.call @create_exception() : () -> !py.dynamic
	cf.cond_br %arg0, ^throw, ^normal_return

^throw:
	py.raise %0

^normal_return:
	return %0 : !py.dynamic
}

func @__init__() -> !py.dynamic {
	%0 = test.random
	%1 = py.call @inline_foo(%0) : (i1) -> !py.dynamic
	test.use(%1) : !py.dynamic
	%2 = py.invoke @inline_foo(%0) : (i1) -> !py.dynamic
		label ^continue unwind ^retException

^continue:
	return %2 : !py.dynamic

^retException(%e : !py.dynamic):
	return %e : !py.dynamic
}

// CHECK-LABEL: @__init__
// CHECK-NEXT: %[[RANDOM:.*]] = test.random
// CHECK-NEXT: %[[EX:.*]] = py.call @create_exception()
// CHECK-NEXT: cf.cond_br %[[RANDOM]], ^[[THROW:.*]], ^[[CONTINUE:[[:alnum:]]+]]
// CHECK-NEXT: ^[[THROW]]:
// CHECK-NEXT: py.raise %[[EX]]
// CHECK-NEXT: ^[[CONTINUE]]:
// CHECK-NEXT: cf.br ^[[CONTINUE:.*]](
// CHECK-SAME: %[[EX]]
// CHECK-NEXT: ^[[CONTINUE]]
// CHECK-SAME: %[[EX:[[:alnum:]]+]]
// CHECK-NEXT: test.use(%[[EX]])
// CHECK-NEXT: %[[EX:.*]] = py.invoke @create_exception()
// CHECK-NEXT: label ^[[SUCCESS:[[:alnum:]]+]] unwind ^[[HANDLER:[[:alnum:]]+]]
// CHECK-NEXT: ^[[SUCCESS]]
// CHECK-NEXT: cf.cond_br %[[RANDOM]], ^[[THROW:.*]], ^[[CONTINUE:[[:alnum:]]+]]
// CHECK-NEXT: ^[[THROW]]:
// CHECK-NEXT: cf.br ^[[HANDLER:.*]](
// CHECK-SAME: %[[EX]]
// CHECK-NEXT: ^[[CONTINUE]]
// CHECK-NEXT: cf.br ^[[CONTINUE:.*]](
// CHECK-SAME: %[[EX]]
// CHECK-NEXT: ^[[CONTINUE]]
// CHECK-SAME: %[[EX:[[:alnum:]]+]]
// CHECK-NEXT: cf.br ^[[CONTINUE:[[:alnum:]]+]]
// CHECK-NEXT: ^[[CONTINUE]]:
// CHECK-NEXT: return %[[EX]]
// CHECK-NEXT: ^[[HANDLER]](%[[EX:[[:alnum:]]+]]: {{.*}}):
// CHECK-NEXT: return %[[EX]]
