// RUN: pylir-opt %s --test-inliner-interface --split-input-file | FileCheck %s
// RUN: pylir-opt %s --test-inliner-interface --split-input-file -mlir-print-debuginfo -mlir-print-local-scope  | FileCheck %s --check-prefix INLINE-LOC

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.BaseException = #py.type

func.func private @create_exception() -> !py.dynamic

func.func @inline_foo(%arg0 : i1) -> !py.dynamic {
	%0 = py.call @create_exception() : () -> !py.dynamic
	cf.cond_br %arg0, ^throw, ^normal_return

^throw:
	py.raise %0

^normal_return:
	return %0 : !py.dynamic
}

func.func @__init__() -> !py.dynamic {
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

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.BaseException = #py.type
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.str = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.function = #py.type
py.globalValue const @builtins.None = #py.type

py.globalValue "private" @function

func.func @inline_foo(%arg0 : i1) -> !py.dynamic {
    %f = py.constant(@function)
	%0 = py.function.call %f()
	cf.cond_br %arg0, ^throw, ^normal_return

^throw:
	py.raise %0

^normal_return:
	return %0 : !py.dynamic
}

func.func @__init__() -> !py.dynamic {
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
// CHECK-NEXT: %[[F:.*]] = py.constant(@function)
// CHECK-NEXT: %[[EX:.*]] = py.function.call %[[F]]()
// CHECK-NEXT: cf.cond_br %[[RANDOM]], ^[[THROW:.*]], ^[[CONTINUE:[[:alnum:]]+]]
// CHECK-NEXT: ^[[THROW]]:
// CHECK-NEXT: py.raise %[[EX]]
// CHECK-NEXT: ^[[CONTINUE]]:
// CHECK-NEXT: cf.br ^[[CONTINUE:.*]](
// CHECK-SAME: %[[EX]]
// CHECK-NEXT: ^[[CONTINUE]]
// CHECK-SAME: %[[EX:[[:alnum:]]+]]
// CHECK-NEXT: test.use(%[[EX]])
// CHECK-NEXT: %[[F:.*]] = py.constant(@function)
// CHECK-NEXT: %[[EX:.*]] = py.function.invoke %[[F]]()
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

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.BaseException = #py.type
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.str = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.function = #py.type
py.globalValue const @builtins.None = #py.type

py.globalValue "private" @function

func.func @inline_foo() -> !py.dynamic {
    %0 = py.constant(@builtins.BaseException) loc("source.mlir":146:69)
    %1 = py.makeObject %0 loc("source.mlir":147:49)
	py.raise %1
}

func.func @test_loc() -> !py.dynamic {
	%1 = py.call @inline_foo() : () -> !py.dynamic loc("source.mlir":152:74)
	return %1 : !py.dynamic
}

// INLINE-LOC-LABEL: @test_loc
// INLINE-LOC-NEXT: %[[TYPE:.*]] = py.constant(@builtins.BaseException) loc(callsite("source.mlir":146:69 at "source.mlir":152:74))
// INLINE-LOC-NEXT: %[[EX:.*]] = py.makeObject %[[TYPE]] loc(callsite("source.mlir":147:49 at "source.mlir":152:74))
// INLINE-LOC-NEXT: py.raise %[[EX]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.BaseException = #py.type
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.str = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.function = #py.type
py.globalValue const @builtins.None = #py.type

py.globalValue "private" @function

func.func @inline_foo() -> !py.dynamic {
    %0 = py.constant(@builtins.BaseException)
    %1 = py.makeObject %0
	py.raise %1
}

func.func @__init__() -> !py.dynamic {
	%1 = py.call @inline_foo() : () -> !py.dynamic
    %r = test.random
    cf.cond_br %r, ^bb0, ^bb1

^bb0:
	return %1 : !py.dynamic

^bb1:
    test.use(%r) : i1
    return %1 : !py.dynamic
}

// CHECK-LABEL: @__init__
// CHECK-NEXT: %[[TYPE:.*]] = py.constant(@builtins.BaseException)
// CHECK-NEXT: %[[EX:.*]] = py.makeObject %[[TYPE]]
// CHECK-NEXT: py.raise %[[EX]]
