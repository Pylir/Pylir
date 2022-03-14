// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

// Stubs
py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.bool = #py.type<>

// CHECK-LABEL: @same_value
// CHECK: %[[RES:.*]] = py.constant #py.bool<value = True>
// CHECK: return %[[RES]]
func @same_value() -> !py.dynamic {
    %0 = py.makeTuple ()
    %1 = py.is %0, %0
    %2 = py.bool.fromI1 %1
    return %2 : !py.dynamic
}

// -----

// Stubs
py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.bool = #py.type<>

// CHECK-LABEL: @two_allocs
// CHECK: %[[RES:.*]] = py.constant #py.bool<value = False>
// CHECK: return %[[RES]]
func @two_allocs(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeTuple (%arg0)
    %1 = py.makeTuple (%arg0)
    %2 = py.is %0, %1
    %3 = py.bool.fromI1 %2
    return %3 : !py.dynamic
}

// -----

// Stubs
py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.bool = #py.type<>

// CHECK-LABEL: @singletons
// CHECK: %[[RES:.*]] = py.constant #py.bool<value = True>
// CHECK: return %[[RES]]
func @singletons() -> !py.dynamic {
    %0 = py.constant @builtins.bool
    %1 = py.constant @builtins.bool
    %2 = py.is %0, %1
    %3 = py.bool.fromI1 %2
    return %3 : !py.dynamic
}

// -----

// Stubs
py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.bool = #py.type<>

// CHECK-LABEL: @alloca_symbol
// CHECK: %[[RES:.*]] = py.constant #py.bool<value = False>
// CHECK: return %[[RES]]
func @alloca_symbol(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant @builtins.bool
    %1 = py.makeTuple (%arg0)
    %2 = py.is %0, %1
    %3 = py.bool.fromI1 %2
    return %3 : !py.dynamic
}
