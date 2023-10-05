// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

#builtins_tuple = #py.globalValue<builtins.tuple, initializer = #py.type>

// CHECK: #[[$TUPLE:.*]] = #py.globalValue<builtins.tuple{{.*}}>

py.func @test() -> !py.dynamic {
    %0 = constant(#py.tuple<(#builtins_tuple)>)
    %1 = arith.constant 0 : index
    %2 = tuple_getItem %0[%1]
    return %2 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-DAG: %[[C1:.*]] = constant(#[[$TUPLE]])
// CHECK: return %[[C1]]

py.func @test2(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = constant(#builtins_tuple)
    %1 = makeTuple (%0, * %arg0)
    %2 = arith.constant 0 : index
    %3 = tuple_getItem %1[%2]
    return %3 : !py.dynamic
}

// CHECK-LABEL: @test2
// CHECK-DAG: %[[C1:.*]] = constant(#[[$TUPLE]])
// CHECK: return %[[C1]]

py.func @test3(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = constant(#builtins_tuple)
    %1 = tuple_prepend %0, %arg0
    %2 = arith.constant 0 : index
    %3 = tuple_getItem %1[%2]
    return %3 : !py.dynamic
}

// CHECK-LABEL: @test3
// CHECK-DAG: %[[C1:.*]] = constant(#[[$TUPLE]])
// CHECK: return %[[C1]]
