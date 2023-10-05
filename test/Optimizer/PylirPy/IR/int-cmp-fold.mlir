// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.func @test1(%arg0 : !py.dynamic) -> i1 {
    %0 = constant(#py.int<5>)
    %1 = int_cmp eq %0, %arg0
    return %1 : i1
}

// CHECK-LABEL: @test1
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = constant(#py.int<5>)
// CHECK-NEXT: %[[RESULT:.*]] = int_cmp eq %[[ARG0]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]

py.func @test2(%arg0 : !py.dynamic) -> i1 {
    %0 = constant(#py.int<5>)
    %1 = int_cmp ne %0, %arg0
    return %1 : i1
}

// CHECK-LABEL: @test2
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = constant(#py.int<5>)
// CHECK-NEXT: %[[RESULT:.*]] = int_cmp ne %[[ARG0]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]

py.func @test3(%arg0 : !py.dynamic) -> i1 {
    %0 = constant(#py.int<5>)
    %1 = arith.constant 1 : i1
    %2 = int_cmp ne %0, %arg0
    %3 = arith.xori %1, %2 : i1
    return %3 : i1
}

// CHECK-LABEL: @test3
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = constant(#py.int<5>)
// CHECK-NEXT: %[[RESULT:.*]] = int_cmp eq %[[ARG0]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]

py.func @test4(%arg0 : !py.dynamic) -> i1 {
    %0 = constant(#py.int<5>)
    %1 = arith.constant 1 : i1
    %2 = int_cmp eq %0, %arg0
    %3 = arith.xori %1, %2 : i1
    return %3 : i1
}

// CHECK-LABEL: @test4
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = constant(#py.int<5>)
// CHECK-NEXT: %[[RESULT:.*]] = int_cmp ne %[[ARG0]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]

py.func @test5(%arg0 : !py.dynamic) -> i1 {
    %0 = constant(#py.int<5>)
    %1 = arith.constant 1 : i1
    %2 = int_cmp lt %0, %arg0
    %3 = arith.xori %1, %2 : i1
    return %3 : i1
}

// CHECK-LABEL: @test5
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = constant(#py.int<5>)
// CHECK-NEXT: %[[RESULT:.*]] = int_cmp le %[[ARG0]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]


py.func @test6(%arg0 : !py.dynamic) -> i1 {
    %0 = constant(#py.int<5>)
    %1 = arith.constant 1 : i1
    %2 = int_cmp le %0, %arg0
    %3 = arith.xori %1, %2 : i1
    return %3 : i1
}

// CHECK-LABEL: @test6
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = constant(#py.int<5>)
// CHECK-NEXT: %[[RESULT:.*]] = int_cmp lt %[[ARG0]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]

py.func @test7(%arg0 : !py.dynamic) -> i1 {
    %0 = constant(#py.int<5>)
    %1 = arith.constant 1 : i1
    %2 = int_cmp gt %0, %arg0
    %3 = arith.xori %1, %2 : i1
    return %3 : i1
}

// CHECK-LABEL: @test7
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = constant(#py.int<5>)
// CHECK-NEXT: %[[RESULT:.*]] = int_cmp ge %[[ARG0]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]


py.func @test8(%arg0 : !py.dynamic) -> i1 {
    %0 = constant(#py.int<5>)
    %1 = arith.constant 1 : i1
    %2 = int_cmp ge %0, %arg0
    %3 = arith.xori %1, %2 : i1
    return %3 : i1
}

// CHECK-LABEL: @test8
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-NEXT: %[[C:.*]] = constant(#py.int<5>)
// CHECK-NEXT: %[[RESULT:.*]] = int_cmp gt %[[ARG0]], %[[C]]
// CHECK-NEXT: return %[[RESULT]]

py.func @test_ne_lt() -> i1 {
    %n = arith.constant true
    %0 = constant(#py.int<1>)
    %1 = constant(#py.int<2>)
    %2 = int_cmp eq %0, %1
    %3 = int_cmp ne %0, %1
    %4 = int_cmp lt %0, %1
    %5 = int_cmp le %0, %1
    %6 = int_cmp gt %0, %1
    %7 = int_cmp ge %0, %1
    %8 = arith.xori %2, %n : i1
    %9 = arith.xori %6, %n : i1
    %10 = arith.xori %7, %n : i1
    %12 = arith.andi %3, %8 : i1
    %13 = arith.andi %12, %4 : i1
    %14 = arith.andi %13, %5 : i1
    %15 = arith.andi %14, %9 : i1
    %16 = arith.andi %15, %10 : i1
    return %16 : i1
}

// CHECK-LABEL: py.func @test_ne_lt
// CHECK-NEXT: %[[C:.*]] = arith.constant true
// CHECK-NEXT: return %[[C]]

py.func @test_ne_gt() -> i1 {
    %n = arith.constant true
    %0 = constant(#py.int<2>)
    %1 = constant(#py.int<1>)
    %2 = int_cmp eq %0, %1
    %3 = int_cmp ne %0, %1
    %4 = int_cmp lt %0, %1
    %5 = int_cmp le %0, %1
    %6 = int_cmp gt %0, %1
    %7 = int_cmp ge %0, %1
    %8 = arith.xori %2, %n : i1
    %9 = arith.xori %4, %n : i1
    %10 = arith.xori %5, %n : i1
    %12 = arith.andi %3, %8 : i1
    %13 = arith.andi %12, %6 : i1
    %14 = arith.andi %13, %7 : i1
    %15 = arith.andi %14, %9 : i1
    %16 = arith.andi %15, %10 : i1
    return %16 : i1
}

// CHECK-LABEL: py.func @test_ne_gt
// CHECK-NEXT: %[[C:.*]] = arith.constant true
// CHECK-NEXT: return %[[C]]

py.func @test_eq() -> i1 {
    %n = arith.constant true
    %0 = constant(#py.bool<True>)
    %1 = constant(#py.int<1>)
    %2 = int_cmp eq %0, %1
    %3 = int_cmp ne %0, %1
    %4 = int_cmp lt %0, %1
    %5 = int_cmp le %0, %1
    %6 = int_cmp gt %0, %1
    %7 = int_cmp ge %0, %1
    %8 = arith.xori %3, %n : i1
    %9 = arith.xori %4, %n : i1
    %10 = arith.xori %6, %n : i1
    %12 = arith.andi %2, %8 : i1
    %13 = arith.andi %12, %9 : i1
    %14 = arith.andi %13, %5 : i1
    %15 = arith.andi %14, %10 : i1
    %16 = arith.andi %15, %7 : i1
    return %16 : i1
}

// CHECK-LABEL: py.func @test_eq
// CHECK-NEXT: %[[C:.*]] = arith.constant true
// CHECK-NEXT: return %[[C]]

// -----

// CHECK-LABEL: @test_redundant_convert_1
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]: index
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]: index
// CHECK: %[[R:.*]] = arith.cmpi eq, %[[ARG0]], %[[ARG1]]
// CHECK: return %[[R]]
py.func @test_redundant_convert_1(%arg0: index, %arg1: index) -> i1 {
    %0 = int_fromUnsigned %arg0
    %1 = int_fromUnsigned %arg1
    %2 = int_cmp eq %0, %1
    return %2 : i1
}

// CHECK-LABEL: @test_redundant_convert_2
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]: index
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]: index
// CHECK: %[[R:.*]] = arith.cmpi slt, %[[ARG0]], %[[ARG1]]
// CHECK: return %[[R]]
py.func @test_redundant_convert_2(%arg0: index, %arg1: index) -> i1 {
    %0 = int_fromUnsigned %arg0
    %1 = int_fromSigned %arg1
    %2 = int_cmp lt %0, %1
    return %2 : i1
}

// CHECK-LABEL: @test_redundant_convert_3
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]: index
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]: index
// CHECK: %[[R:.*]] = arith.cmpi sgt, %[[ARG0]], %[[ARG1]]
// CHECK: return %[[R]]
py.func @test_redundant_convert_3(%arg0: index, %arg1: index) -> i1 {
    %0 = int_fromSigned %arg0
    %1 = int_fromUnsigned %arg1
    %2 = int_cmp gt %0, %1
    return %2 : i1
}

// CHECK-LABEL: @test_redundant_convert_4
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]: index
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]: index
// CHECK: %[[R:.*]] = arith.cmpi ne, %[[ARG0]], %[[ARG1]]
// CHECK: return %[[R]]
py.func @test_redundant_convert_4(%arg0: index, %arg1: index) -> i1 {
    %0 = int_fromSigned %arg0
    %1 = int_fromSigned %arg1
    %2 = int_cmp ne %0, %1
    return %2 : i1
}

// CHECK-LABEL: @test_redundant_convert_constant
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]: index
// CHECK: %[[C:.*]] = arith.constant 5
// CHECK: %[[R:.*]] = arith.cmpi sgt, %[[ARG0]], %[[C]]
// CHECK: return %[[R]]
py.func @test_redundant_convert_constant(%arg0: index) -> i1 {
    %0 = int_fromSigned %arg0
    %1 = constant(#py.int<5>)
    %2 = int_cmp gt %0, %1
    return %2 : i1
}

// CHECK-LABEL: @test_redundant_convert_constant_too_large
// CHECK-NOT: arith.cmpi
py.func @test_redundant_convert_constant_too_large(%arg0: index) -> i1 {
    %0 = int_fromSigned %arg0
    %1 = constant(#py.int<5596967597659764578954876548654865457694675736657365763575676576584678>)
    %2 = int_cmp gt %0, %1
    return %2 : i1
}
