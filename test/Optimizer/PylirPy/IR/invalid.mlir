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

py.globalValue const @builtins.type = #py.type<mro_tuple = #py.tuple<(#py.ref<@builtins.type>, #py.ref<@builtins.object>)>>
py.globalValue const @builtins.object = #py.type<mro_tuple = #py.tuple<(#py.ref<@builtins.object>)>>
py.globalValue const @builtins.function = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.str = #py.type
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.None = #py.type

func.func private @foo(!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic

py.globalValue const @Foo = #py.type<mro_tuple = #py.tuple<(#py.ref<@Foo>, #py.ref<@builtins.object>)>, slots = {
    __hash__ = #py.function<@foo>
}>

py.globalValue @bar = #py.obj<#py.ref<@Foo>>
// expected-error@below {{Constant dictionary not allowed to have key whose type's '__hash__' method is not off of a builtin.}}
py.globalValue @dict = #py.dict<{#py.ref<@bar> to #py.ref<@builtins.None>}>

// -----

py.globalValue const @builtins.type = #py.type<mro_tuple = #py.tuple<(#py.ref<@builtins.type>, #py.ref<@builtins.object>)>>
py.globalValue const @builtins.object = #py.type<mro_tuple = #py.tuple<(#py.ref<@builtins.object>)>>
py.globalValue const @builtins.function = #py.type
py.globalValue const @builtins.dict = #py.type
py.globalValue const @builtins.str = #py.type
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.None = #py.type

func.func private @foo(!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic

py.globalValue @Foo = #py.type<mro_tuple = #py.tuple<(#py.ref<@Foo>, #py.ref<@builtins.object>)>>

py.globalValue @bar = #py.obj<#py.ref<@Foo>>
// expected-error@below {{Constant dictionary not allowed to have key whose type's '__hash__' method is not off of a builtin.}}
py.globalValue @dict = #py.dict<{#py.ref<@bar> to #py.ref<@builtins.None>}>

// -----

// expected-error@below {{Expected integer attribute initializer}}
py.global @lol1 : index = 5.3 : f64

// -----

// expected-error@below {{Expected float attribute initializer}}
py.global @lol2 : f64 = 5 : index

// -----

// expected-error@below {{Expected initializer of type 'ObjectAttrInterface' or 'RefAttr' to global value}}
py.global @lol3 : !py.dynamic = 5 : index

// -----

py.global @lol : !py.dynamic

// expected-error@below {{RefAttr '@lol' does not refer to a 'py.globalValue'}}
py.global @lol4 : !py.dynamic = #py.ref<@lol>

// -----

// expected-error@below {{RefAttr '@lol' does not refer to a 'py.globalValue'}}
py.global @lol5: !py.dynamic = #py.ref<@lol>

// -----

func.func @foo() {
    // expected-error@below {{RefAttr '@lol' does not refer to a 'py.globalValue'}}
    %0 = py.constant(#py.ref<@lol>)
    return
}

// -----

py.global @lol : index = 5 : index

func.func @foo() {
    // expected-error@below {{RefAttr '@lol' does not refer to a 'py.globalValue'}}
    %0 = py.constant(#py.ref<@lol>)
    return
}

// -----

// expected-error@below {{Expected MRO to refer to a tuple}}
py.globalValue @builtins.type = #py.type<mro_tuple = #py.int<5>>
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.tuple = #py.type

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.None = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.function = #py.type

func.func private @foo(!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic

// expected-error@below {{Expected __defaults__ to refer to a tuple}}
py.globalValue @lol = #py.function<@foo, defaults = #py.int<5>>

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.None = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.function = #py.type

func.func private @foo(!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic

// expected-error@below {{Expected __kwdefaults__ to refer to a dictionary}}
py.globalValue @lol = #py.function<@foo, kw_defaults = #py.int<5>>

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.None = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.function = #py.type

func.func private @foo(!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic

// expected-error@below {{Expected __dict__ to refer to a dictionary}}
py.globalValue @lol = #py.function<@foo, dict = #py.int<5>>

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type

py.globalValue @lol = #py.int<5>

// expected-error@below {{Incorrect normalized key entry '#py.ref<@lol>' for key-value pair '(#py.ref<@lol>, #py.int<3>)'}}
py.globalValue @foo = #py.dict<{#py.ref<@lol> to #py.int<3>}>

// -----

// expected-error@below {{Expected 'instance_slots' to refer to a tuple of strings}}
py.globalValue const @builtins.type = #py.type<instance_slots = <(#py.int<5>)>>
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type