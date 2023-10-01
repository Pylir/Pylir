// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str
#builtins_float = #py.globalValue<builtins.float, initializer = #py.type>
py.external @builtins.float, #builtins_float
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

py.func @constant_tuple() -> index {
    %0 = constant(#py.tuple<(#py.int<0>, #py.str<"text">, #py.float<5.0>)>)
    %1 = tuple_len %0
    return %1 : index
}

// CHECK-LABEL: @constant_tuple
// CHECK: %[[RESULT:.*]] = arith.constant 3 : index
// CHECK: return %[[RESULT]]

// -----

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str
#builtins_float = #py.globalValue<builtins.float, initializer = #py.type>
py.external @builtins.float, #builtins_float
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

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

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple

py.func @make_tuple(%arg0 : !py.dynamic, %arg1 : !py.dynamic) -> index {
    %0 = makeTuple (%arg0, %arg1)
    %1 = tuple_len %0
    return %1 : index
}

// CHECK-LABEL: @make_tuple
// CHECK: %[[RESULT:.*]] = arith.constant 2 : index
// CHECK: return %[[RESULT]]
