// RUN: pylir-opt --test-linker %s | FileCheck %s

module {
func.func private @foo(i32)

func.func private @bar(i32)
}

module {
func.func @foo(%arg0 : i32) {
    test.use(%arg0) : i32
    return
}

func.func private @bar(i32)
}

// CHECK: module {
// CHECK-NEXT: module {
// CHECK-NEXT: func.func private @bar(i32)
// CHECK-NEXT: func.func @foo(%[[ARG0:.*]]: i32) {
// CHECK-NEXT: test.use(%[[ARG0]]) : i32
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
