// RUN: pylir-opt %s | pylir-opt | FileCheck %s

// CHECK-LABEL: pyHIR.globalFunc @test(
// CHECK-SAME: %{{[[:alnum:]]+}},
// CHECK-SAME: *%[[ARG1:[[:alnum:]]+]],
// CHECK-SAME: %{{.*}} "keyword",
// CHECK-SAME: %{{.*}} {test.name = 0 : i32}) -> !py.dynamic {
// CHECK-NEXT: %{{.*}} = func "foo"(%{{.*}} "rest" = %[[ARG1]]) -> !py.dynamic {

pyHIR.globalFunc @test(%arg0, *%arg1, %arg2 "keyword", %arg3 { test.name = 0 : i32 }) -> !py.dynamic {
    %0 = func "foo"(%ff0 "rest" = %arg1) -> !py.dynamic {
        return %ff0
    }
    return %0
}

// CHECK-LABEL: pyHIR.globalFunc @res_attr()
// CHECK-SAME: -> (!py.dynamic {test.name = 0 : i32})
pyHIR.globalFunc @res_attr() -> (!py.dynamic { test.name = 0 : i32 }) {
    %0 = func "foo"(%ff0 "rest") -> !py.dynamic {
        return %ff0
    }
    return %0
}

// CHECK-LABEL: pyHIR.init "__main__" {
// CHECK: init_return
pyHIR.init "__main__" {
    %0 = func "foo"(%ff0 "rest") {
        return
    }
    init_return %0
}
