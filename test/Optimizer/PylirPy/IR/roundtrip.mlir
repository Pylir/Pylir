// RUN: pylir-opt %s | pylir-opt | FileCheck %s

func @bar() {
    return
}

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.bool = #py.type
py.globalValue @builtins.BaseException = #py.type

// CHECK-LABEL: func @foo
func @foo() -> !py.unknown {
    %0 = py.constant(#py.bool<True>) : !py.unknown
    py.invoke @bar() : () -> ()
        label ^happy unwind ^exception(%0 : !py.unknown)

^happy:
    %1 = py.constant(#py.bool<False>) : !py.unknown
    return %1 : !py.unknown

^exception(%2 : !py.unknown):
    // CHECK: %[[EXCEPTION:.*]] = py.landingPad @builtins.BaseException
    // CHECK-NEXT: py.br ^[[BLOCK:[[:alnum:]]+]]
    // CHECK-SAME: %[[EXCEPTION]]
    // CHECK-SAME-1: %{{[[:alnum:]]+}}
    %3 = py.landingPad @builtins.BaseException : !py.unknown
    py.br ^baseExcept(%3, %2 : !py.unknown, !py.unknown)

// CHECK: ^[[BLOCK]]
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK: return %[[ARG1]]
^baseExcept(%e : !py.unknown, %4 : !py.unknown):
    return %4 : !py.unknown
}
