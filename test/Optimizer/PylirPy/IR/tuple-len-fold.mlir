// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.func @constant_tuple() -> index {
    %0 = constant(#py.tuple<(#py.int<0>, #py.str<"text">, #py.float<5.0>)>)
    %1 = tuple_len %0
    return %1 : index
}

// CHECK-LABEL: @constant_tuple
// CHECK: %[[RESULT:.*]] = arith.constant 3 : index
// CHECK: return %[[RESULT]]

// -----

#foo = #py.globalValue<foo, initializer = #py.tuple<(#py.int<0>, #py.str<"text">, #py.float<5.0>)>>

py.func @constant_tuple() -> index {
    %0 = constant(#foo)
    %1 = tuple_len %0
    return %1 : index
}

// CHECK-LABEL: @constant_tuple
// CHECK: %[[RESULT:.*]] = arith.constant 3 : index
// CHECK: return %[[RESULT]]

// -----

py.func @make_tuple(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> index {
    %0 = makeTuple (%arg0, %arg1)
    %1 = tuple_len %0
    return %1 : index
}

// CHECK-LABEL: @make_tuple
// CHECK: %[[RESULT:.*]] = arith.constant 2 : index
// CHECK: return %[[RESULT]]
