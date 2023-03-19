// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.float = #py.type
py.globalValue @builtins.bool = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.tuple = #py.type

py.globalValue @g1 = #py.int<3>
py.globalValue @g2 = #py.float<5.0>
py.globalValue @g3 = #py.bool<True>
py.globalValue @g4 = #py.str<"text">
py.globalValue @g5 = #py.tuple<(#py.ref<@g1>, #py.ref<@g2>, #py.ref<@g3>, #py.ref<@g4>)>

// CHECK: globalValue const @g1
// CHECK: globalValue const @g2
// CHECK: globalValue const @g3
// CHECK: globalValue const @g4
// CHECK: globalValue const @g5
