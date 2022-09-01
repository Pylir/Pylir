// RUN: pylir-opt %s -split-input-file -verify-diagnostics

func.func @test() {
	py.call @foo() : () -> () // expected-error {{'py.call' op failed to find function named '@foo'}}
	return
}

// -----

func.func @test() {
	%0 = arith.constant true
	py.call @test(%0) : (i1) -> () // expected-error {{call operand types are not compatible with argument types of '@test'}}
	return
}

// -----

py.globalValue const @builtins.type = #py.type<mroTuple = #py.tuple<(@builtins.type, @builtins.object)>>
py.globalValue const @builtins.object = #py.type<mroTuple = #py.tuple<(@builtins.object)>>
py.globalValue const @builtins.function = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.str = #py.type
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.None = #py.type

func.func private @foo(!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic

py.globalValue const @Foo = #py.type<mroTuple = #py.tuple<(@Foo, @builtins.object)>, slots = {
    __hash__ = #py.function<@foo>
}>

py.globalValue @bar = #py.obj<@Foo>
// expected-error@below {{Constant dictionary not allowed to have key whose type's '__hash__' method is not off of a builtin.}}
py.globalValue @dict = #py.dict<{@bar to @builtins.None}>

// -----

py.globalValue const @builtins.type = #py.type<mroTuple = #py.tuple<(@builtins.type, @builtins.object)>>
py.globalValue const @builtins.object = #py.type<mroTuple = #py.tuple<(@builtins.object)>>
py.globalValue const @builtins.function = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.str = #py.type
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.None = #py.type

func.func private @foo(!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic

py.globalValue @Foo = #py.type<mroTuple = #py.tuple<(@Foo, @builtins.object)>>

py.globalValue @bar = #py.obj<@Foo>
// expected-error@below {{Constant dictionary not allowed to have key whose type's '__hash__' method is not off of a builtin.}}
py.globalValue @dict = #py.dict<{@bar to @builtins.None}>
