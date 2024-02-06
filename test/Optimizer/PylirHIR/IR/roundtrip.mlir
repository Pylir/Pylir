// RUN: pylir-opt %s | pylir-opt | FileCheck %s

// CHECK-LABEL: pyHIR.globalFunc @test(
// CHECK-SAME: %{{[[:alnum:]]+}},
// CHECK-SAME: *%[[ARG1:[[:alnum:]]+]],
// CHECK-SAME: %{{.*}} "keyword",
// CHECK-SAME: %{{.*}} {test.name = 0 : i32},
// CHECK-SAME: %{{.*}} only "lol" has_default
// CHECK-NEXT: %{{.*}} = func "foo"(%{{.*}} "rest" = %[[ARG1]]) {

pyHIR.globalFunc @test(%arg0, *%arg1, %arg2 "keyword", %arg3 { test.name = 0 : i32 }, %arg4 only "lol" has_default) {
    %0 = func "foo"(%ff0 "rest" = %arg1) {
        return %ff0
    }
    return %0
}

// CHECK-LABEL: pyHIR.globalFunc @res_attr()
// CHECK-SAME: -> {test.name = 0 : i32}
pyHIR.globalFunc @res_attr() -> { test.name = 0 : i32 } {
    %0 = func "foo"(%ff0 "rest") {
        return %ff0
    }
    return %0
}

// CHECK-LABEL: pyHIR.init "__main__" {
// CHECK: init_return
pyHIR.init "__main__" {
    %0 = func "foo"(%ff0 "rest") {
        return %ff0
    }
    init_return %0
}
