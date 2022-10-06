// RUN: pylir-opt %s -split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.float = #py.type

py.globalValue @foo = #py.dict<{#py.int<5> to #py.int<3>, #py.float<5.0> to #py.int<8>}>

// CHECK-LABEL: @foo
// CHECK-SAME: {
// CHECK-SAME: #py.int<5> to #py.int<8>
// CHECK-NOT: ,
// CHECK-SAME: }
