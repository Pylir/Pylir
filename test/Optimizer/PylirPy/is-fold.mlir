// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL: @same_value
// CHECK: %[[RES:.*]] = py.constant #py.bool<True>
// CHECK: return %[[RES]]
func @same_value() -> !py.dynamic {
    %0 = py.makeTuple ()
    %1 = py.is %0, %0
    return %1 : !py.dynamic
}

// -----

// CHECK-LABEL: @two_allocs
// CHECK: %[[RES:.*]] = py.constant #py.bool<False>
// CHECK: return %[[RES]]
func @two_allocs(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.makeTuple (%arg0)
    %1 = py.makeTuple (%arg0)
    %2 = py.is %0, %1
    return %2 : !py.dynamic
}

// -----

// CHECK-LABEL: @singletons
// CHECK: %[[RES:.*]] = py.constant #py.bool<True>
// CHECK: return %[[RES]]
func @singletons() -> !py.dynamic {
    %0 = py.singleton None
    %1 = py.singleton None
    %2 = py.is %0, %1
    return %2 : !py.dynamic
}

