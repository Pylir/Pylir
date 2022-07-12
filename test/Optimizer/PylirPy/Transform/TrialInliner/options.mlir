// RUN: pylir-opt %s --pylir-trial-inliner='optimization-pipeline=test-hello-world' | FileCheck %s
// CHECK: Hello World!

func.func @foo() {
    return
}

func.func @main() {
    py.call @foo() : () -> ()
    return
}
