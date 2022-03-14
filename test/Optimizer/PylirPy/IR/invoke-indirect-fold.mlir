// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type<>
py.globalValue @builtins.BaseException = #py.type<>

func private @foo()

func @test() {
    %0 = constant @foo : () -> ()
    py.invoke_indirect %0() : () -> ()
        label ^bb0 unwind ^bb1

^bb0:
    return

^bb1:
    %1 = py.landingPad @builtins.BaseException
    py.landingPad.br ^handler

^handler:
    return
}

// CHECK-LABEL: @test
// CHECK: py.invoke @foo() : () -> ()

// CHECK: py.landingPad @builtins.BaseException
// CHECK-NEXT: py.landingPad.br
