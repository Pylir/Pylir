// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

// Stubs
py.globalValue @builtins.type = #py.type
py.globalValue @builtins.bool = #py.type

// CHECK-LABEL: @same_value
// CHECK: %[[RES:.*]] = arith.constant true
// CHECK: return %[[RES]]
func @same_value() -> i1 {
    %0 = py.makeTuple () : () -> !py.unknown
    %1 = py.is %0, %0 : !py.unknown, !py.unknown
    return %1 : i1
}

// -----

// Stubs
py.globalValue @builtins.type = #py.type
py.globalValue @builtins.bool = #py.type

// CHECK-LABEL: @two_allocs
// CHECK: %[[RES:.*]] = arith.constant false
// CHECK: return %[[RES]]
func @two_allocs(%arg0 : !py.unknown) -> i1 {
    %0 = py.makeTuple (%arg0) : (!py.unknown) -> !py.unknown
    %1 = py.makeTuple (%arg0) : (!py.unknown) -> !py.unknown
    %2 = py.is %0, %1 : !py.unknown, !py.unknown
    return %2 : i1
}

// -----

// Stubs
py.globalValue @builtins.type = #py.type
py.globalValue @builtins.bool = #py.type

// CHECK-LABEL: @singletons
// CHECK: %[[RES:.*]] = arith.constant true
// CHECK: return %[[RES]]
func @singletons() -> i1 {
    %0 = py.constant(@builtins.bool) : !py.unknown
    %1 = py.constant(@builtins.bool) : !py.unknown
    %2 = py.is %0, %1 : !py.unknown, !py.unknown
    return %2 : i1
}

// -----

// Stubs
py.globalValue @builtins.type = #py.type
py.globalValue @builtins.bool = #py.type

// CHECK-LABEL: @alloca_symbol
// CHECK: %[[RES:.*]] = arith.constant false
// CHECK: return %[[RES]]
func @alloca_symbol(%arg0 : !py.unknown) -> i1 {
    %0 = py.constant(@builtins.bool) : !py.unknown
    %1 = py.makeTuple (%arg0) : (!py.unknown) -> !py.unknown
    %2 = py.is %0, %1 : !py.unknown, !py.unknown
    return %2 : i1
}
