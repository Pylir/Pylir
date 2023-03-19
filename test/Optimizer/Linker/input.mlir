// RUN: pylir-opt --test-linker %s | FileCheck %s

module {
py.func private @foo(i32)

py.func private @bar(i32)
}

module {
py.func @foo(%arg0 : i32) {
    test.use(%arg0) : i32
    return
}

py.func private @bar(i32)
}

// CHECK: module {
// CHECK-NEXT: module {
// CHECK-NEXT: py.func private @bar(i32)
// CHECK-NEXT: py.func @foo(%[[ARG0:.*]]: i32) {
// CHECK-NEXT: test.use(%[[ARG0]]) : i32
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
