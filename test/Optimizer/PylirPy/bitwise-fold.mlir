// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL: @and_ints
// CHECK: %[[RES:.*]] = py.constant #py.int<1>
// CHECK: return %[[RES]]
func @and_ints() -> !py.dynamic {
    %0 = py.constant #py.int<3>
    %1 = py.constant #py.int<5>
    %2 = py.and %0, %1
    return %2 : !py.dynamic
}

// -----

// CHECK-LABEL: @or_ints
// CHECK: %[[RES:.*]] = py.constant #py.int<7>
// CHECK: return %[[RES]]
func @or_ints() -> !py.dynamic {
    %0 = py.constant #py.int<3>
    %1 = py.constant #py.int<5>
    %2 = py.or %0, %1
    return %2 : !py.dynamic
}

// -----

// CHECK-LABEL: @xor_ints
// CHECK: %[[RES:.*]] = py.constant #py.int<6>
// CHECK: return %[[RES]]
func @xor_ints() -> !py.dynamic {
    %0 = py.constant #py.int<3>
    %1 = py.constant #py.int<5>
    %2 = py.xor %0, %1
    return %2 : !py.dynamic
}

// -----

// CHECK-LABEL: @and_bools
// CHECK: %[[RES:.*]] = py.constant #py.bool<False>
// CHECK: return %[[RES]]
func @and_bools() -> !py.dynamic {
    %0 = py.constant #py.bool<True>
    %1 = py.constant #py.bool<False>
    %2 = py.and %0, %1
    return %2 : !py.dynamic
}

// -----

// CHECK-LABEL: @or_bools
// CHECK: %[[RES:.*]] = py.constant #py.bool<True>
// CHECK: return %[[RES]]
func @or_bools() -> !py.dynamic {
    %0 = py.constant #py.bool<True>
    %1 = py.constant #py.bool<False>
    %2 = py.or %0, %1
    return %2 : !py.dynamic
}

// -----

// CHECK-LABEL: @xor_bools
// CHECK: %[[RES:.*]] = py.constant #py.bool<True>
// CHECK: return %[[RES]]
func @xor_bools() -> !py.dynamic {
    %0 = py.constant #py.bool<True>
    %1 = py.constant #py.bool<False>
    %2 = py.xor %0, %1
    return %2 : !py.dynamic
}
