// RUN: pylir-opt %s -split-input-file -verify-diagnostics

// expected-error@below {{only one positional rest parameter allowed}}
pyHIR.globalFunc @two_pos_rest(*%arg0, *%arg1) {
  return
}

// -----

// expected-error@below {{only one keyword rest parameter allowed}}
pyHIR.globalFunc @two_keyword_rest(**%arg0, **%arg1) {
  return
}

// -----

pyHIR.globalFunc @call_kind_arguments_mismatch(%arg0) {
  // expected-error@below {{"kind_internal" must be the same size as argument operands}}
  %0 = "pyHIR.call"(%arg0, %arg0)
    <{ kind_internal = array<i32>, keywords = ["rest"] }>
    : (!py.dynamic, !py.dynamic) -> !py.dynamic
  return %0
}

// -----

pyHIR.globalFunc @call_invalid_kind_value(%arg0) {
  // expected-error@below {{invalid value 4 in "kind_internal" array}}
  %0 = "pyHIR.call"(%arg0, %arg0)
    <{ kind_internal = array<i32: 4>, keywords = ["rest"] }>
    : (!py.dynamic, !py.dynamic) -> !py.dynamic
  return %0
}

// -----

pyHIR.globalFunc @call_invalid_kind_index(%arg0) {
  // expected-error@below {{out-of-bounds index 1 into keywords array with size 1}}
  %0 = "pyHIR.call"(%arg0, %arg0)
    <{ kind_internal = array<i32: -1>, keywords = ["rest"] }>
    : (!py.dynamic, !py.dynamic) -> !py.dynamic
  return %0
}

// -----

pyHIR.init "__main__" {
  %0 = py.constant(#py.dict<{}>)
  // expected-error@below {{cannot initialize '__main__' module}}
  initModule @__main__
  init_return
}

// -----

// expected-error@below {{expected at least one parameter (the closure parameter) to be present}}
pyHIR.globalFunc @no_closure_param() {
  %0 = py.constant(#py.int<3>)
  return %0
}


// -----

// expected-error@below {{closure parameter must be positional-only with no default}}
pyHIR.globalFunc @wrong_closure_param_1(%closure "text") {
  %0 = py.constant(#py.int<3>)
  return %0
}

// -----

// expected-error@below {{closure parameter must be positional-only with no default}}
pyHIR.globalFunc @wrong_closure_param_2(%closure has_default) {
  %0 = py.constant(#py.int<3>)
  return %0
}

// -----

pyHIR.globalFunc @wrong_block_args1(%closure) {
  // expected-error@below {{expected entry block of pyHIR.class to have exactly one py.dynamic argument}}
  %0 = class "test" {
    class_return
  }
  return %0
}
