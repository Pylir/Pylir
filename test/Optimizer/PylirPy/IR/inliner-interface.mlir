// RUN: pylir-opt %s --test-inliner-interface | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.BaseException = #py.type

func private @create_exception() -> !py.unknown

func @inline_foo(%arg0 : i1) -> !py.unknown {
	%0 = py.call @create_exception() : () -> !py.unknown
	py.cond_br %arg0, ^throw, ^normal_return

^throw:
	py.raise %0 : !py.unknown

^normal_return:
	py.return %0 : !py.unknown
}

func @__init__() -> !py.unknown {
	%0 = test.random
	%1 = py.call @inline_foo(%0) : (i1) -> !py.unknown
	test.use(%1) : !py.unknown
	%2 = py.invoke @inline_foo(%0) : (i1) -> !py.unknown
		label ^continue unwind ^failure

^continue:
	py.return %2 : !py.unknown

^failure:
	%3 = py.landingPad @builtins.BaseException : !py.unknown
	py.br ^retException(%3 : !py.unknown)

^retException(%e : !py.unknown):
	py.return %e : !py.unknown
}

// CHECK-LABEL: @__init__
// CHECK-NEXT: %[[RANDOM:.*]] = test.random
// CHECK-NEXT: %[[EX:.*]] = py.call @create_exception() : () -> !py.unknown
// CHECK-NEXT: py.cond_br %[[RANDOM]], ^[[THROW:.*]], ^[[CONTINUE:[[:alnum:]]+]]
// CHECK-NEXT: ^[[THROW]]:
// CHECK-NEXT: py.raise %[[EX]]
// CHECK-NEXT: ^[[CONTINUE]]:
// CHECK-NEXT: py.br ^[[CONTINUE:.*]](
// CHECK-SAME: %[[EX]]
// CHECK-NEXT: ^[[CONTINUE]]
// CHECK-SAME: %[[EX:[[:alnum:]]+]]
// CHECK-NEXT: test.use(%[[EX]])
// CHECK-NEXT: %[[EX:.*]] = py.call @create_exception()
// CHECK-NEXT: py.cond_br %[[RANDOM]], ^[[THROW:.*]], ^[[CONTINUE:[[:alnum:]]+]]
// CHECK-NEXT: ^[[THROW]]:
// CHECK-NEXT: py.br ^[[HANDLER:.*]](
// CHECK-SAME: %[[EX]]
// CHECK-NEXT: ^[[CONTINUE]]
// CHECK-NEXT: py.br ^[[CONTINUE:.*]](
// CHECK-SAME: %[[EX]]
// CHECK-NEXT: ^[[CONTINUE]]
// CHECK-SAME: %[[EX:[[:alnum:]]+]]
// CHECK-NEXT: py.br ^[[CONTINUE:[[:alnum:]]+]]
// CHECK-NEXT: ^[[CONTINUE]]:
// CHECK-NEXT: py.return %[[EX]]
// CHECK: ^[[HANDLER]](%[[EX:[[:alnum:]]+]]: {{.*}}):
// CHECK-NEXT: py.return %[[EX]]
