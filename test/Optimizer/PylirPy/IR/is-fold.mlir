// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

// Stubs
py.globalValue @builtins.type = #py.type
py.globalValue @builtins.bool = #py.type

// CHECK-LABEL: @same_value
// CHECK: %[[RES:.*]] = arith.constant true
// CHECK: return %[[RES]]
func.func @same_value() -> i1 {
    %0 = py.makeTuple ()
    %1 = py.is %0, %0
    return %1 : i1
}

// -----

// Stubs
py.globalValue @builtins.type = #py.type
py.globalValue @builtins.bool = #py.type

// CHECK-LABEL: @two_allocs
// CHECK: %[[RES:.*]] = arith.constant false
// CHECK: return %[[RES]]
func.func @two_allocs(%arg0 : !py.dynamic) -> i1 {
    %0 = py.makeTuple (%arg0)
    %1 = py.makeTuple (%arg0)
    %2 = py.is %0, %1
    return %2 : i1
}

// -----

// Stubs
py.globalValue @builtins.type = #py.type
py.globalValue @builtins.bool = #py.type

// CHECK-LABEL: @singletons
// CHECK: %[[RES:.*]] = arith.constant true
// CHECK: return %[[RES]]
func.func @singletons() -> i1 {
    %0 = py.constant(@builtins.bool)
    %1 = py.constant(@builtins.bool)
    %2 = py.is %0, %1
    return %2 : i1
}

// -----

// Stubs
py.globalValue @builtins.type = #py.type
py.globalValue @builtins.bool = #py.type

// CHECK-LABEL: @alloca_symbol
// CHECK: %[[RES:.*]] = arith.constant false
// CHECK: return %[[RES]]
func.func @alloca_symbol(%arg0 : !py.dynamic) -> i1 {
    %0 = py.constant(@builtins.bool)
    %1 = py.makeTuple (%arg0)
    %2 = py.is %0, %1
    return %2 : i1
}
