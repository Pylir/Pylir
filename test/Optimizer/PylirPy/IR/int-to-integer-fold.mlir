// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.int = #py.type

func @test() -> (i64, i1) {
    %0 = py.constant #py.int<5>
    %1, %valid = py.int.toInteger %0 : i64
    return %1, %valid : i64, i1
}

// CHECK-LABEL: @test
// CHECK-DAG: %[[C1:.*]] = arith.constant 5
// CHECK-DAG: %[[C2:.*]] = arith.constant true
// CHECK-NEXT: return %[[C1]], %[[C2]]


func @test2() -> (i64, i1) {
    %0 = py.constant #py.int<534567452345234523154235234523463462345345234523937628376129387126381253128735>
    %1, %valid = py.int.toInteger %0 : i64
    return %1, %valid : i64, i1
}

// CHECK-LABEL: @test2
// CHECK: %[[C2:.*]] = arith.constant false
// CHECK: return %{{.*}}, %[[C2]]

func @test3(%arg0 : i64) -> (i32, i1) {
    %0 = py.int.fromInteger %arg0 : i64
    %1, %2 = py.int.toInteger %0 : i32
    return %1, %2 : i32, i1
}

// CHECK-LABEL: @test3
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-DAG: %[[C1:.*]] = arith.constant true
// CHECK-DAG: %[[RESULT:.*]] = arith.trunci %[[ARG0]] : i64 to i32
// CHECK: return %[[RESULT]], %[[C1]]

func @test4(%arg0 : i32) -> (i64, i1) {
    %0 = py.int.fromInteger %arg0 : i32
    %1, %2 = py.int.toInteger %0 : i64
    return %1, %2 : i64, i1
}

// CHECK-LABEL: @test4
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-DAG: %[[C1:.*]] = arith.constant true
// CHECK-DAG: %[[RESULT:.*]] = arith.extsi %[[ARG0]] : i32 to i64
// CHECK: return %[[RESULT]], %[[C1]]

func @test5(%arg0 : i64) -> (i64, i1) {
    %0 = py.int.fromInteger %arg0 : i64
    %1, %2 = py.int.toInteger %0 : i64
    return %1, %2 : i64, i1
}

// CHECK-LABEL: @test5
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-DAG: %[[C1:.*]] = arith.constant true
// CHECK: return %[[ARG0]], %[[C1]]

func @test6(%arg0 : index) -> (i64, i1) {
    %0 = py.int.fromInteger %arg0 : index
    %1, %2 = py.int.toInteger %0 : i64
    return %1, %2 : i64, i1
}

// CHECK-LABEL: @test6
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-DAG: %[[C1:.*]] = arith.constant true
// CHECK-DAG: %[[RESULT:.*]] = arith.index_cast %[[ARG0]] : index to i64
// CHECK: return %[[RESULT]], %[[C1]]

func @test7(%arg0 : i64) -> (index, i1) {
    %0 = py.int.fromInteger %arg0 : i64
    %1, %2 = py.int.toInteger %0 : index
    return %1, %2 : index, i1
}

// CHECK-LABEL: @test7
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-DAG: %[[C1:.*]] = arith.constant true
// CHECK-DAG: %[[RESULT:.*]] = arith.index_cast %[[ARG0]] : i64 to index
// CHECK: return %[[RESULT]], %[[C1]]

func @test8() -> (index, i1) {
    %0 = py.constant #py.int<5>
    %1, %valid = py.int.toInteger %0 : index
    return %1, %valid : index, i1
}

// CHECK-LABEL: @test8
// CHECK-DAG: %[[C1:.*]] = arith.constant 5 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant true
// CHECK-NEXT: return %[[C1]], %[[C2]]
