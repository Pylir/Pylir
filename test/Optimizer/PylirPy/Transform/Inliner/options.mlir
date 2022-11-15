// RUN: pylir-opt %s --pylir-inliner='optimization-pipeline=any(test-hello-world) max-inlining-iterations=1' | FileCheck %s
// CHECK-COUNT-2: Hello World!

func.func @foo() {
    return
}

func.func @main() {
    py.call @foo() : () -> ()
    return
}
