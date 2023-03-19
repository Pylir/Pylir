// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

// Stubs
py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.bool = #py.type

// CHECK-LABEL: @same_value
// CHECK: %[[RES:.*]] = arith.constant true
// CHECK: return %[[RES]]
py.func @same_value() -> i1 {
    %0 = makeTuple ()
    %1 = is %0, %0
    return %1 : i1
}

// -----

// Stubs
py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.bool = #py.type

// CHECK-LABEL: @two_allocs
// CHECK: %[[RES:.*]] = arith.constant false
// CHECK: return %[[RES]]
py.func @two_allocs(%arg0 : !py.dynamic) -> i1 {
    %0 = makeTuple (%arg0)
    %1 = makeTuple (%arg0)
    %2 = is %0, %1
    return %2 : i1
}

// -----

// Stubs
py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.bool = #py.type

// CHECK-LABEL: @singletons
// CHECK: %[[RES:.*]] = arith.constant true
// CHECK: return %[[RES]]
py.func @singletons() -> i1 {
    %0 = constant(#py.ref<@builtins.bool>)
    %1 = constant(#py.ref<@builtins.bool>)
    %2 = is %0, %1
    return %2 : i1
}

// CHECK-LABEL: @singletons_not
// CHECK: %[[RES:.*]] = arith.constant false
// CHECK: return %[[RES]]
py.func @singletons_not() -> i1 {
    %0 = constant(#py.ref<@builtins.bool>)
    %1 = constant(#py.ref<@builtins.tuple>)
    %2 = is %0, %1
    return %2 : i1
}

// -----

// Stubs
py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.bool = #py.type

// CHECK-LABEL: @alloca_symbol
// CHECK: %[[RES:.*]] = arith.constant false
// CHECK: return %[[RES]]
py.func @alloca_symbol(%arg0 : !py.dynamic) -> i1 {
    %0 = constant(#py.ref<@builtins.bool>)
    %1 = makeTuple (%arg0)
    %2 = is %0, %1
    %3 = is %1, %0
    %4 = arith.ori %2, %3 : i1
    return %4 : i1
}
