// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_dict= #py.globalValue<builtins.dict, initializer = #py.type>
py.external @builtins.dict, #builtins_dict
#builtins_int = #py.globalValue<builtins.int, initializer = #py.type>
py.external @builtins.int, #builtins_int
#builtins_float = #py.globalValue<builtins.float, initializer = #py.type>
py.external @builtins.float, #builtins_float
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

py.func @test(%arg0 : !py.dynamic, %arg1 : index) -> i1 {
    %0 = constant(#py.dict<{}>)
    %2 = dict_tryGetItem %0[%arg0 hash(%arg1)]
    %3 = isUnboundValue %2
    return %3 : i1
}

// CHECK-LABEL: @test
// CHECK-DAG: %[[C1:.*]] = arith.constant true
// CHECK: return %[[C1]]

py.func @test2(%arg0 : index) -> !py.dynamic {
    %0 = constant(#py.dict<{#py.int<5> to #py.str<"value">}>)
    %1 = constant(#py.float<5.0>)
    %2 = dict_tryGetItem %0[%1 hash(%arg0)]
    return %2 : !py.dynamic
}

// CHECK-LABEL: @test2
// CHECK: %[[C:.*]] = constant(#py.str<"value">)
// CHECK: return %[[C]]
