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

// expected-error@below {{Expected initializer of type 'ObjectAttrInterface' or 'GlobalValueAttr' to global value}}
py.global @lol3 : !py.dynamic = 5 : index

// -----

py.func @test() -> !py.dynamic {
  // expected-error@below {{Expected MRO to refer to a tuple}}
  %0 = constant(#py.type<mro_tuple = #py.int<5>>)
  return %0 : !py.dynamic
}

// -----

py.func private @foo(!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic

py.func @test() -> !py.dynamic {
  // expected-error@below {{Expected __defaults__ to refer to a tuple}}
  %0 = constant(#py.function<@foo, defaults = #py.int<5>>)
  return %0 : !py.dynamic
}

// -----

py.func private @foo(!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic

py.func @test() -> !py.dynamic {
  // expected-error@below {{Expected __kwdefaults__ to refer to a dictionary}}
  %0 = constant(#py.function<@foo, kw_defaults = #py.int<5>>)
  return %0 : !py.dynamic
}

// -----

py.func private @foo(!py.dynamic, !py.dynamic, !py.dynamic) -> !py.dynamic

py.func @test() -> !py.dynamic {
  // expected-error@below {{Expected __dict__ to refer to a dictionary}}
  %0 = constant(#py.function<@foo, dict = #py.int<5>>)
  return %0 : !py.dynamic
}

// -----

py.func @test() -> !py.dynamic {
  // expected-error@below {{Expected 'instance_slots' to refer to a tuple of strings}}
  %0 = constant(#py.type<instance_slots = <(#py.int<5>)>>)
  return %0 : !py.dynamic
}

// -----

py.func private @foo() -> !py.dynamic

py.func @test() {
  // expected-error@below {{call result types '' are not compatible with output types '!py.dynamic' of '@foo'}}
  call @foo() : () -> ()
  return
}

// -----

py.func @func_get_closure(%arg0 : !py.dynamic) -> i32 {
  // expected-error@below {{index '2' out of bounds}}
  %0 = "py.function_closureArg"(%arg0) <{ index = 2 : i32, closure_types = [i32, i64]}> : (!py.dynamic) -> i32
  return %0 : i32
}
