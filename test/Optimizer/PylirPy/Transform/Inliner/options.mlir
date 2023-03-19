// RUN: pylir-opt %s --pylir-inliner='optimization-pipeline=any(test-hello-world) max-inlining-iterations=1' | FileCheck %s
// CHECK-COUNT-2: Hello World!

py.func @foo() {
    return
}

py.func @main() {
    call @foo() : () -> ()
    return
}
