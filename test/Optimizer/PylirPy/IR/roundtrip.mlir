// RUN: pylir-opt %s | pylir-opt | FileCheck %s

func @bar() {
    return
}

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.bool = #py.type<>
py.globalValue @builtins.BaseException = #py.type<>

// CHECK-LABEL: func @foo
func @foo() -> !py.dynamic {
    %0 = py.constant #py.bool<value = True>
    py.invoke @bar() : () -> ()
        label ^happy unwind ^exception(%0)

^happy:
    %1 = py.constant #py.bool<value = False>
    return %1 : !py.dynamic

^exception(%2 : !py.dynamic):
    // CHECK: %[[EXCEPTION:.*]] = py.landingPad @builtins.BaseException
    // CHECK-NEXT: py.landingPad.br ^[[BLOCK:[[:alnum:]]+]]
    // CHECK-SAME: %[[EXCEPTION]]
    // CHECK-SAME-1: %{{[[:alnum:]]+}}
    %3 = py.landingPad @builtins.BaseException
    py.landingPad.br ^baseExcept(%3, %2)

// CHECK: ^[[BLOCK]]
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK: return %[[ARG1]]
^baseExcept(%e : !py.dynamic, %4 : !py.dynamic):
    return %4 : !py.dynamic
}

// CHECK-LABEL: func @type_switch
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-DAG: %[[TYPE:.*]] = py.constant @builtins.type
// CHECK-DAG: %[[BOOL:.*]] = py.constant @builtins.bool
// CHECK: %[[RESULT:.*]] = py.typeSwitch %[[ARG0]] {
// CHECK-NEXT: %[[GENERIC:.*]] = py.typeOf %[[ARG0]]
// CHECK-NEXT: py.yield %[[GENERIC]]
// CHECK-NEXT: } case %[[TYPE]] {
// CHECK-NEXT: py.yield %[[TYPE]]
// CHECK-NEXT: } case %[[BOOL]] {
// CHECK-NEXT: py.yield %[[TYPE]]
// CHECK: return %[[RESULT]]

func @type_switch(%arg0 : !py.dynamic) -> !py.dynamic {
    %0 = py.constant @builtins.type
    %1 = py.constant @builtins.bool
    %2 = py.typeSwitch %arg0 {
        %3 = py.typeOf %arg0
        py.yield %3 : !py.dynamic
    } case %0 {
        py.yield %0 : !py.dynamic
    } case %1 {
        py.yield %0 : !py.dynamic
    } : !py.dynamic
    return %2 : !py.dynamic
}
