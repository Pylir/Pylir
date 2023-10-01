// RUN: pylir-opt %s -convert-pylirPy-to-pylirMem --split-input-file | FileCheck %s

// CHECK: #[[$STR_TYPE:.*]] = #py.globalValue<builtins.str>
#builtins_str = #py.globalValue<builtins.str>

py.func @test(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = constant(#builtins_str)
    %1 = str_copy %arg0 : %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[STR:.*]] = constant(#[[$STR_TYPE]])
// CHECK-NEXT: %[[SLOTS:.*]] = type_slots %[[STR]]
// CHECK-NEXT: %[[LEN:.*]] = tuple_len %[[SLOTS]]
// CHECK-NEXT: %[[MEMORY:.*]] = pyMem.gcAllocObject %[[STR]][%[[LEN]]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initStr %[[MEMORY]] to %[[ARG0]]
// CHECK-NEXT: return %[[RESULT]]
