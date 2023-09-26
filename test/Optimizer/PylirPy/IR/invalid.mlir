// RUN: pylir-opt %s -split-input-file -verify-diagnostics

py.func @test() {
	py.call @foo() : () -> () // expected-error {{'py.call' op failed to find function named '@foo'}}
	return
}

// -----

py.func @test() {
	%0 = arith.constant true
	py.call @test(%0) : (i1) -> () // expected-error {{call operand types are not compatible with argument types of '@test'}}
	return
}

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

py.func @foo() {
    // expected-error@below {{RefAttr '@lol' does not refer to a 'py.globalValue'}}
    %0 = constant(#py.ref<@lol>)
    return
}

// -----

py.global @lol : index = 5 : index

py.func @foo() {
    // expected-error@below {{RefAttr '@lol' does not refer to a 'py.globalValue'}}
    %0 = constant(#py.ref<@lol>)
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

py.func private @foo(!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic

// expected-error@below {{Expected __defaults__ to refer to a tuple}}
py.globalValue @lol = #py.function<@foo, defaults = #py.int<5>>

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.None = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.function = #py.type

py.func private @foo(!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic

// expected-error@below {{Expected __kwdefaults__ to refer to a dictionary}}
py.globalValue @lol = #py.function<@foo, kw_defaults = #py.int<5>>

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.None = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.function = #py.type

py.func private @foo(!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic

// expected-error@below {{Expected __dict__ to refer to a dictionary}}
py.globalValue @lol = #py.function<@foo, dict = #py.int<5>>

// -----

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type

py.globalValue @lol = #py.int<5>

// expected-error@below {{invalid kind of attribute specified}}
py.globalValue @foo = #py.dict<{#py.ref<@lol> to #py.int<3>}>

// -----

// expected-error@below {{Expected 'instance_slots' to refer to a tuple of strings}}
py.globalValue const @builtins.type = #py.type<instance_slots = <(#py.int<5>)>>
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type
