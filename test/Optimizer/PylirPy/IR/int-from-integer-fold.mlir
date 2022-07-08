// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func.func @test() -> !py.dynamic {
    %0 = arith.constant 5 : i64
    %1 = py.int.fromInteger %0 : i64
    return %1 : !py.dynamic
}

// CHECK-LABEL: @test
// CHECK-NEXT: %[[C:.*]] = py.constant(#py.int<5>)
// CHECK-NEXT: return %[[C]]
