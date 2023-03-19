// RUN: pylir-opt %s --pylir-monomorph --split-input-file | FileCheck %s


py.globalValue @builtins.type = #py.type
py.globalValue @builtins.tuple = #py.type
py.globalValue @builtins.int = #py.type
py.globalValue @builtins.str = #py.type
py.globalValue @builtins.dict = #py.type
py.globalValue @builtins.function = #py.type
py.globalValue @builtins.None = #py.type

py.func @real(%arg0 : !py.dynamic, %arg1 : !py.dynamic, %arg2 : !py.dynamic) -> !py.dynamic {
    %0 = typeOf %arg0
    return %0 : !py.dynamic
}

py.globalValue @function = #py.function<@real>

py.func @__init__() -> !py.dynamic {
    %0 = constant(#py.ref<@function>)
    %1 = py.function.call %0(%0, %0, %0)
    return %1 : !py.dynamic
}

// CHECK-LABEL: py.func @real

// CHECK-LABEL: @__init__
// CHECK: call @[[REAL_CLONE:.*]](%{{.*}})

// CHECK: py.func private @[[REAL_CLONE:([[:alnum:]]|_)+]]
// CHECK: %[[TYPE:.*]] = constant(#py.ref<@builtins.function>)
// CHECK: return %[[TYPE]]
