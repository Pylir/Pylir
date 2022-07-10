// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func.func @real(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.typeOf %arg0
    return %0 : !py.dynamic
}

func.func @stub(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.call @real(%arg0) : (!py.dynamic) -> !py.dynamic
    return %0 : !py.dynamic
}

func.func @__init__() -> !py.dynamic {
    %0 = py.constant(#py.int<0>)
    %1 = py.call @stub(%0) : (!py.dynamic) -> !py.dynamic
    return %1 : !py.dynamic
}

// CHECK-LABEL: func.func @real

// CHECK-LABEL: func.func @stub
// CHECK: py.call @real(

// CHECK-LABEL: @__init__
// CHECK: py.call @[[STUB_CLONE:.*]](%{{.*}})

// CHECK: func.func private @[[REAL_CLONE:.*]](%{{.*}}: !py.dynamic)
// CHECK: %[[TYPE:.*]] = py.constant(@builtins.int)
// CHECK: return %[[TYPE]]

// CHECK: func.func private @[[STUB_CLONE]]
// CHECK: py.call @[[REAL_CLONE]](%{{.*}})
