// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue const @tuple = #py.tuple<(#py.str<"__slots__">)>
py.globalValue const @builtins.type = #py.type<slots = {__slots__ = @tuple}>
py.globalValue const @builtins.tuple = #py.type
py.globalValue const @builtins.str = #py.type

func.func @test1() -> !py.dynamic {
    %0 = py.constant(#py.tuple<(@builtins.type)>)
    %1 = py.mroLookup "__slots__" in %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: func @test1
// CHECK: %[[C1:.*]] = py.constant(@tuple)
// CHECK: return %[[C1]]

func.func @test2() -> !py.dynamic {
    %0 = py.constant(#py.tuple<()>)
    %1 = py.mroLookup "__slots__" in %0
    return %1 : !py.dynamic
}

// CHECK-LABEL: func @test2
// CHECK: %[[C1:.*]] = py.constant(#py.unbound)
// CHECK: return %[[C1]]

func.func @test3(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant(@builtins.type)
    %1 = py.makeTuple (%0, %arg0)
    %2 = py.mroLookup "__slots__" in %1
    return %2 : !py.dynamic
}

// CHECK-LABEL: func @test3
// CHECK-DAG: %[[C1:.*]] = py.constant(@tuple)
// CHECK: return %[[C1]]
