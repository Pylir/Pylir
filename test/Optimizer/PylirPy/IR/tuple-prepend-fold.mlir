// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

#builtins_type = #py.globalValue<builtins.type, initializer = #py.type>
py.external @builtins.type, #builtins_type
#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>
py.external @builtins.tuple, #builtins_tuple
#builtins_str = #py.globalValue<builtins.str, initializer = #py.type>
py.external @builtins.str, #builtins_str

py.func @test() -> !py.dynamic {
    %0 = constant(#py.tuple<()>)
    %1 = constant(#builtins_tuple)
    %2 = tuple_prepend %1, %0
    return %2 : !py.dynamic
}

// CHECK: #[[$TUPLE:.*]] = #py.globalValue<builtins.tuple{{.*}}>

// CHECK-LABEL: @test
// CHECK: %[[C:.*]] = constant(#py.tuple<(#[[$TUPLE]])>)
// CHECK: return %[[C]]

py.func @test3(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = constant(#py.tuple<(#py.str<"value">)>)
    %2 = tuple_prepend %arg0, %0
    return %2 : !py.dynamic
}

// CHECK-LABEL: @test3
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-DAG: %[[C:.*]] = constant(#py.str<"value">)
// CHECK: %[[RESULT:.*]] = makeTuple (%[[ARG0]], %[[C]])
// CHECK: return %[[RESULT]]
