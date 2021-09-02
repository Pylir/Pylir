// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL: @pow_floats
// CHECK: %[[RES:.*]] = py.constant 1.25{{0*}}e+{{0*}}2
// CHECK: return %[[RES]]
func @pow_floats() -> !py.dynamic {
    %0 = py.constant 5.0
    %1 = py.constant 3.0
    %2 = py.pow %0 to %1
    return %2 : !py.dynamic
}

// -----

// CHECK-LABEL: @pow_neg_expo
// CHECK: %[[RES:.*]] = py.constant 1.250000e-01 : f64
// CHECK: return %[[RES]]
func @pow_neg_expo() -> !py.dynamic {
    %0 = py.constant #py.int<2>
    %1 = py.constant #py.int<-3>
    %2 = py.pow %0 to %1
    return %2 : !py.dynamic
}

// -----

// CHECK-LABEL: @pow_base_one
// CHECK: %[[RES:.*]] = py.constant #py.int<1>
// CHECK: return %[[RES]]
func @pow_base_one() -> !py.dynamic {
    %0 = py.constant #py.int<1>
    %1 = py.constant #py.int<342342345245234>
    %2 = py.pow %0 to %1
    return %2 : !py.dynamic
}

// -----

// CHECK-LABEL: @pow_expo_one
// CHECK: %[[RES:.*]] = py.constant #py.int<342342345245234>
// CHECK: return %[[RES]]
func @pow_expo_one() -> !py.dynamic {
    %0 = py.constant #py.int<342342345245234>
    %1 = py.constant #py.int<1>
    %2 = py.pow %0 to %1
    return %2 : !py.dynamic
}

// -----

// CHECK-LABEL: @pow_integer
// CHECK: %[[RES:.*]] = py.constant #py.int<125>
// CHECK: return %[[RES]]
func @pow_integer() -> !py.dynamic {
    %0 = py.constant #py.int<5>
    %1 = py.constant #py.int<3>
    %2 = py.pow %0 to %1
    return %2 : !py.dynamic
}
