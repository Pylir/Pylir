// RUN: pylir-opt %s -canonicalize --split-input-file | FileCheck %s

py.globalValue @builtins.type = #py.type
py.globalValue @builtins.str = #py.type
py.globalHandle @foo

func.func @test() -> !py.dynamic {
  %0 = py.constant(#py.str<"value">)
  py.store %0 into @foo
  cf.br ^bb1(%0 : !py.dynamic)

^bb1(%1: !py.dynamic):
  %2 = test.random
  cf.cond_br %2, ^bb1(%1 : !py.dynamic), ^bb2

^bb2:
  return %1 : !py.dynamic
}

// CHECK-LABEL: func.func @test
// CHECK: %[[C:.*]] = py.constant(#py.str<"value">)
// CHECK: cf.br ^[[BB1:[[:alnum:]]+]]
// CHECK-NOT: (

// CHECK: ^[[BB1]]:
// CHECK: cf.cond_br %{{.*}}, ^[[BB1]], ^[[BB2:[[:alnum:]]+]]

// CHECK: ^[[BB2]]:
// CHECK: return %[[C]]
