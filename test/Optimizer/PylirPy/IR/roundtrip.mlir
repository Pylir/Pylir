// RUN: pylir-opt %s | pylir-opt | FileCheck %s

func @bar() {
    return
}

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.bool = #py.type
py.globalValue @builtins.BaseException = #py.type

// CHECK-LABEL: func @foo
func @foo() -> !py.dynamic {
    %0 = py.constant #py.bool<True>
    py.invoke @bar() : () -> ()
        label ^happy unwind ^exception(%0)

^happy:
    %1 = py.constant #py.bool<False>
    return %1 : !py.dynamic

^exception(%2 : !py.dynamic):
    // CHECK: %[[EXCEPTION:.*]] = py.landingPad @builtins.BaseException
    // CHECK-NEXT: cf.br ^[[BLOCK:[[:alnum:]]+]]
    // CHECK-SAME: %[[EXCEPTION]]
    // CHECK-SAME-1: %{{[[:alnum:]]+}}
    %3 = py.landingPad @builtins.BaseException
    cf.br ^baseExcept(%3, %2 : !py.dynamic, !py.dynamic)

// CHECK: ^[[BLOCK]]
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK: return %[[ARG1]]
^baseExcept(%e : !py.dynamic, %4 : !py.dynamic):
    return %4 : !py.dynamic
}
