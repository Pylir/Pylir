// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue const @tuple = #py.tuple<(#py.str<"__slots__">)>
py.globalValue const @builtins.type = #py.type<slots: #py.slots<{"__slots__" to @tuple}>>
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.str = #py.type

func @test1() -> (!py.dynamic, i1) {
    %0 = py.constant #py.tuple<(@builtins.type)>
    %1, %found = py.mroLookup "__slots__" in %0
    return %1, %found : !py.dynamic, i1
}

// CHECK-LABEL: func @test1
// CHECK-DAG: %[[C1:.*]] = py.constant @tuple
// CHECK-DAG: %[[C2:.*]] = arith.constant true
// CHECK: return %[[C1]], %[[C2]]

func @test2() -> (!py.dynamic, i1) {
    %0 = py.constant #py.tuple<()>
    %1, %found = py.mroLookup "__slots__" in %0
    return %1, %found : !py.dynamic, i1
}

// CHECK-LABEL: func @test2
// CHECK-DAG: %[[C1:.*]] = py.constant #py.unbound
// CHECK-DAG: %[[C2:.*]] = arith.constant false
// CHECK: return %[[C1]], %[[C2]]
