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
