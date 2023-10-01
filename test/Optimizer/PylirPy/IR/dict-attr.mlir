// RUN: pylir-opt %s -split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_dict= #py.globalValue<builtins.dict, initializer = #py.type>
py.external @builtins.dict, #builtins_dict
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int
#builtins_float = #py.globalValue<builtins.float, initializer = #py.type>
py.external @builtins.float, #builtins_float

// CHECK-LABEL: @foo
py.func @foo() {
  // CHECK-NEXT: constant
  // CHECK-SAME: {
  // CHECK-SAME: #py.int<5> to #py.int<8>
  // CHECK-NOT: ,
  // CHECK-SAME: }
  %0 = constant(#py.dict<{#py.int<5> to #py.int<3>, #py.float<5.0> to #py.int<8>}>)
  return
}
