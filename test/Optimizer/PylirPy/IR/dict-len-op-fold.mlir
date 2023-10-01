// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_dict= #py.globalValue<builtins.dict, initializer = #py.type>
py.external @builtins.dict, #builtins_dict
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

py.func @test() -> index {
    %0 = constant(#py.dict<{#py.str<"test"> to #builtins_str}>)
    %2 = dict_len %0
    return %2 : index
}

// CHECK-LABEL: @test
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK: return %[[C1]]
