// RUN: pylir-opt %s --test-inliner-interface --split-input-file | FileCheck %s
// RUN: pylir-opt %s --test-inliner-interface --split-input-file -mlir-print-debuginfo -mlir-print-local-scope  | FileCheck %s --check-prefix INLINE-LOC

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.BaseException = #py.type

py.func private @create_exception() -> !py.dynamic

py.func @inline_foo(%arg0 : i1) -> !py.dynamic {
	%0 = call @create_exception() : () -> !py.dynamic
	cf.cond_br %arg0, ^throw, ^normal_return

^throw:
	py.raise %0

^normal_return:
	return %0 : !py.dynamic
}

py.func @__init__() -> !py.dynamic {
	%0 = test.random
	%1 = call @inline_foo(%0) : (i1) -> !py.dynamic
	test.use(%1) : !py.dynamic
	%2 = invoke @inline_foo(%0) : (i1) -> !py.dynamic
		label ^continue unwind ^retException

^continue:
	return %2 : !py.dynamic

^retException(%e : !py.dynamic):
	return %e : !py.dynamic
}

// CHECK-LABEL: @__init__
// CHECK-NEXT: %[[RANDOM:.*]] = test.random
// CHECK-NEXT: %[[EX:.*]] = call @create_exception()
// CHECK-NEXT: cf.cond_br %[[RANDOM]], ^[[THROW:.*]], ^[[CONTINUE:[[:alnum:]]+]]
// CHECK-NEXT: ^[[THROW]]:
// CHECK-NEXT: raise %[[EX]]
// CHECK-NEXT: ^[[CONTINUE]]:
// CHECK-NEXT: cf.br ^[[CONTINUE:.*]](
// CHECK-SAME: %[[EX]]
// CHECK-NEXT: ^[[CONTINUE]]
// CHECK-SAME: %[[EX:[[:alnum:]]+]]
// CHECK-NEXT: test.use(%[[EX]])
// CHECK-NEXT: %[[EX:.*]] = invoke @create_exception()
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

py.func @inline_foo(%arg0 : i1) -> !py.dynamic {
    %f = constant(#py.ref<@function>)
	%0 = function_call %f()
	cf.cond_br %arg0, ^throw, ^normal_return

^throw:
	py.raise %0

^normal_return:
	return %0 : !py.dynamic
}

py.func @__init__() -> !py.dynamic {
	%0 = test.random
	%1 = call @inline_foo(%0) : (i1) -> !py.dynamic
	test.use(%1) : !py.dynamic
	%2 = invoke @inline_foo(%0) : (i1) -> !py.dynamic
		label ^continue unwind ^retException

^continue:
	return %2 : !py.dynamic

^retException(%e : !py.dynamic):
	return %e : !py.dynamic
}

// CHECK-LABEL: @__init__
// CHECK-NEXT: %[[RANDOM:.*]] = test.random
// CHECK-NEXT: %[[F:.*]] = constant(#py.ref<@function>)
// CHECK-NEXT: %[[EX:.*]] = function_call %[[F]]()
// CHECK-NEXT: cf.cond_br %[[RANDOM]], ^[[THROW:.*]], ^[[CONTINUE:[[:alnum:]]+]]
// CHECK-NEXT: ^[[THROW]]:
// CHECK-NEXT: raise %[[EX]]
// CHECK-NEXT: ^[[CONTINUE]]:
// CHECK-NEXT: cf.br ^[[CONTINUE:.*]](
// CHECK-SAME: %[[EX]]
// CHECK-NEXT: ^[[CONTINUE]]
// CHECK-SAME: %[[EX:[[:alnum:]]+]]
// CHECK-NEXT: test.use(%[[EX]])
// CHECK-NEXT: %[[F:.*]] = constant(#py.ref<@function>)
// CHECK-NEXT: %[[EX:.*]] = function_invoke %[[F]]()
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

py.func @inline_foo() -> !py.dynamic {
    %0 = constant(#py.ref<@builtins.BaseException>) loc("source.mlir":146:69)
    %1 = makeObject %0 loc("source.mlir":147:49)
	py.raise %1
}

py.func @test_loc() -> !py.dynamic {
	%1 = call @inline_foo() : () -> !py.dynamic loc("source.mlir":152:74)
	return %1 : !py.dynamic
}

// INLINE-LOC-LABEL: @test_loc
// INLINE-LOC-NEXT: %[[TYPE:.*]] = constant(#py.ref<@builtins.BaseException>) loc(callsite("source.mlir":146:69 at "source.mlir":152:74))
// INLINE-LOC-NEXT: %[[EX:.*]] = makeObject %[[TYPE]] loc(callsite("source.mlir":147:49 at "source.mlir":152:74))
// INLINE-LOC-NEXT: raise %[[EX]]

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.BaseException = #py.type
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.str = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.function = #py.type
py.globalValue const @builtins.None = #py.type

py.globalValue "private" @function

py.func @inline_foo() -> !py.dynamic {
    %0 = constant(#py.ref<@builtins.BaseException>)
    %1 = makeObject %0
	py.raise %1
}

py.func @__init__() -> !py.dynamic {
	%1 = call @inline_foo() : () -> !py.dynamic
    %r = test.random
    cf.cond_br %r, ^bb0, ^bb1

^bb0:
	return %1 : !py.dynamic

^bb1:
    test.use(%r) : i1
    return %1 : !py.dynamic
}

// CHECK-LABEL: @__init__
// CHECK-NEXT: %[[TYPE:.*]] = constant(#py.ref<@builtins.BaseException>)
// CHECK-NEXT: %[[EX:.*]] = makeObject %[[TYPE]]
// CHECK-NEXT: raise %[[EX]]

// -----

py.func @inline_foo() -> !py.dynamic {
	%1 = call @inline_foo() : () -> !py.dynamic
    %r = test.random
    cf.cond_br %r, ^bb0, ^bb1

^bb0:
	return %1 : !py.dynamic

^bb1:
    test.use(%r) : i1
    return %1 : !py.dynamic
}

// CHECK-LABEL: py.func @inline_foo
// CHECK-NEXT: %[[CALL:.*]] = call @inline_foo()
// CHECK-NEXT: %[[R:.*]] = test.random
// CHECK-NEXT: cf.cond_br %[[R]], ^[[BB1:.*]], ^[[BB2:[[:alnum:]]+]]
// CHECK-NEXT: ^[[BB1]]:
// CHECK-NEXT: cf.br ^[[BB3:[[:alnum:]]+]](%[[CALL]] : !py.dynamic)
// CHECK-NEXT: ^[[BB2]]:
// CHECK-NEXT: test.use(%[[R]])
// CHECK-NEXT: cf.br ^[[BB3]](%[[CALL]] : !py.dynamic)
// CHECK-NEXT: ^[[BB3]](%[[ARG:.*]]: !py.dynamic):
// CHECK-NEXT: %[[R:.*]] = test.random
// CHECK-NEXT: cf.cond_br %[[R]], ^[[BB4:.*]], ^[[BB5:[[:alnum:]]+]]
// CHECK-NEXT: ^[[BB4]]:
// CHECK-NEXT: return %[[ARG]]
// CHECK-NEXT: ^[[BB5]]:
// CHECK-NEXT: test.use(%[[R]])
// CHECK-NEXT: return %[[ARG]]
