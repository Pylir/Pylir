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

// CHECK-LABEL: @maketuple_simple
// CHECK: %[[RES:.*]] = constant(#py.tuple
// CHECK-SAME: #py.str<"text">
// CHECK-SAME: #py.int<2>
// CHECK-SAME: #py.float<3.000000e+00>
// CHECK: return %[[RES]]
py.func @maketuple_simple() -> !py.dynamic {
    %0 = constant(#py.str<"text">)
    %1 = constant(#py.int<2>)
    %2 = constant(#py.float<3.0>)
    %3 = makeTuple ( %0, %1, %2 )
    return %3 : !py.dynamic
}

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
#builtins_list= #py.globalValue<builtins.list, initializer = #py.type>
py.external @builtins.list, #builtins_list

// CHECK-LABEL: @maketuple_expansion
// CHECK: %[[RES:.*]] = constant(#py.tuple
// CHECK-SAME: #py.str<"text">
// CHECK-SAME: #py.int<2>
// CHECK-SAME: #py.float<3.000000e+00>
// CHECK: return %[[RES]]
py.func @maketuple_expansion() -> !py.dynamic {
    %0 = constant(#py.str<"text">)
    %1 = constant(#py.list<[#py.int<2>, #py.float<3.0>]>)
    %2 = makeTuple ( %0, *%1 )
    return %2 : !py.dynamic
}
