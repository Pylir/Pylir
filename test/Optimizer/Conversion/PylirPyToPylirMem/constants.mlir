// RUN: pylir-opt %s -convert-pylirPy-to-pylirMem --split-input-file | FileCheck %s

// CHECK-DAG: py.globalValue "private" const @[[CONST:.*]] = #py.tuple<(#py.int<3>)>

func @test() -> !py.dynamic {
    %0 = py.constant #py.tuple<(#py.int<3>)>
    return %0 : !py.dynamic
}

func @test2() -> !py.dynamic {
    %0 = py.constant #py.tuple<(#py.int<3>)>
    return %0 : !py.dynamic
}

// CHECK-LABEL: func @test
// CHECK-NEXT: %[[VALUE:.*]] = py.constant @[[CONST:.*]]
// CHECK-NEXT: return %[[VALUE]]

// CHECK-LABEL: func @test2
// CHECK-NEXT: %[[VALUE:.*]] = py.constant @[[CONST:.*]]
// CHECK-NEXT: return %[[VALUE]]
