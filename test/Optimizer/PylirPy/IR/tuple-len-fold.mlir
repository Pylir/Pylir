// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.float = #py.type
py.globalValue @builtins.tuple = #py.type

func @constant_tuple() -> index {
    %0 = py.constant #py.tuple<(#py.int<0>, #py.str<"text">, #py.float<5.0>)>
    %1 = py.tuple.len %0 : index
    return %1 : index
}

// CHECK-LABEL: @constant_tuple
// CHECK: %[[RESULT:.*]] = arith.constant 3 : index
// CHECK: return %[[RESULT]]

// -----
