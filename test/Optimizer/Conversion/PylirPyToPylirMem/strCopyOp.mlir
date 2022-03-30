// RUN: pylir-opt %s -convert-pylirPy-to-pylirMem --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type

func @test(%arg0 : !py.unknown) -> !py.unknown {
    %0 = py.constant(@builtins.str) : !py.unknown
    %1 = py.str.copy %arg0 : %0 : (!py.unknown, !py.unknown) -> !py.unknown
    return %1 : !py.unknown
}

// CHECK-LABEL: func @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[STR:.*]] = py.constant(@builtins.str)
// CHECK-NEXT: %[[MEMORY:.*]] = pyMem.gcAllocObject %[[STR]]
// CHECK-NEXT: %[[RESULT:.*]] = pyMem.initStr %[[MEMORY]] to %[[ARG0]]
// CHECK-NEXT: return %[[RESULT]]
